#ifndef NABO_FUNCTION_H
#define NABO_FUNCTION_H

#include <stdlib.h>
#include <stdio.h>

#include "nabo/nabo.h"


namespace Nabo {
    template <typename T, int InputDim, int OutputDim> 
        class Function {
            public:
                typedef Eigen::Matrix<T, InputDim, 1> Input;
                typedef Eigen::Matrix<T, OutputDim, 1> Output;
                typedef std::pair<Input,Output> Record;
            protected:
                std::vector<Record> uncompiled_data;
                typedef Eigen::Matrix<T, InputDim, Eigen::Dynamic> InputMatrix;
                typedef Eigen::Matrix<T, OutputDim, Eigen::Dynamic> OutputMatrix;
                InputMatrix *eigenized_input;
                OutputMatrix *eigenized_output;
                NearestNeighbourSearch<T> *knn;
                double weight_sigma;

            public:
                void compile() {
                    if ((knn!=NULL) && (uncompiled_data.empty())) {
                        return;
                    }
                    if (!uncompiled_data.empty()) {
                        delete knn; knn = NULL;
                        InputMatrix* new_input = 
                            new InputMatrix(InputDim,uncompiled_data.size()
                                    + (eigenized_input?eigenized_input->cols():0));
                        OutputMatrix* new_output = 
                            new OutputMatrix(OutputDim,uncompiled_data.size()
                                    + (eigenized_output?eigenized_output->cols():0));
                        if (eigenized_input) {
                            new_input->block(0,0,InputDim,eigenized_input->cols()) = *eigenized_input;
                        }
                        for (size_t i=0;i<uncompiled_data.size();i++) {
                            new_input->block(0,i,InputDim,1) = uncompiled_data[i].first;
                            new_output->block(0,i,OutputDim,1) = uncompiled_data[i].second;
                        }
                        delete eigenized_input; eigenized_input = new_input;
                        delete eigenized_output; eigenized_output = new_output;
                        uncompiled_data.clear();
                    }
                    knn = NearestNeighbourSearch<T>::createKDTreeLinearHeap(*eigenized_input);
                }

                bool is_compiled() const {
                    if ((knn!=NULL) && (uncompiled_data.empty())) {
                        return true;
                    }
                    return false;
                }

                void assert_is_compiled() const {
                    bool CHECK_IF_FUNCTION_IS_COMPILED = is_compiled();
                    assert(CHECK_IF_FUNCTION_IS_COMPILED);
                }

                InputMatrix mergeSupport(const InputMatrix & A, const InputMatrix & B, double epsilon=1e-6) const{
                    NearestNeighbourSearch<T> *lknn;
                    lknn = NearestNeighbourSearch<T>::createKDTreeLinearHeap(A);
                    unsigned int N = B.cols(), Np = A.cols();
                    Eigen::MatrixXi indices(1,N);
                    Eigen::MatrixXf dist2(1,N);
                    knn->knn(B,indices,dist2,1,0,NearestNeighbourSearch<T>::ALLOW_SELF_MATCH,epsilon);
                    for (unsigned int j=0;j<N;j++) {
                        if (indices(j)<0) { Np+=1; }
                    }
                    InputMatrix out(InputDim,Np);
                    Np = A.cols();
                    out.block(0,0,InputDim,Np) = A;
                    for (unsigned int j=0;j<N;j++) {
                        if (indices(j)<0) { 
                            out.block(0,Np,InputDim,1) = B.block(0,j,InputDim,1);
                            Np+=1; 
                        }
                    }
                    delete lknn;
                    return out;
                }

            public:
                Function() : eigenized_input(NULL), eigenized_output(NULL), knn(NULL), weight_sigma(1.0) {}

                Function(const InputMatrix I, const OutputMatrix & O) : knn(NULL), weight_sigma(1.0){
                    eigenized_input = new InputMatrix(I);
                    eigenized_output = new OutputMatrix(O);
                    compile();
                }

                Function(const Function & F) : knn(NULL) {
                    weight_sigma = F.weight_sigma;
                    uncompiled_data = F.uncompiled_data;
                    // Deep copy, no shared pointer on purpose here.
                    eigenized_input = new InputMatrix(*F.eigenized_input);
                    eigenized_output = new OutputMatrix(*F.eigenized_output);
                    compile();
                }

                ~Function() {clear();}

                void clear() {
                    uncompiled_data.clear();
                    delete knn; knn = NULL;
                    delete eigenized_input; eigenized_input = NULL;
                    delete eigenized_output; eigenized_output = NULL;
                }

                bool empty() const {
                    return uncompiled_data.empty() && (knn==NULL);
                }

                size_t size() const {
                    return uncompiled_data.size() + (eigenized_input?eigenized_input->cols():0);
                }

                double getWeightSigma() const {return weight_sigma;}
                void setWeightSigma(double w) {weight_sigma = w;}

                void set(const Input & I, const Output & O) {
                    uncompiled_data.push_back(Record(I,O));
                }

                OutputMatrix evalAt(const InputMatrix & I) const {
                    assert_is_compiled();
                    const unsigned int K = 4;
                    unsigned int N = I.cols();
                    Eigen::MatrixXi indices(K,N);
                    Eigen::MatrixXf dist2(K,N);
                    knn->knn(I,indices,dist2,K,0,NNSearchF::SORT_RESULTS | NearestNeighbourSearch<T>::ALLOW_SELF_MATCH);

                    OutputMatrix out(OutputDim,N);
                    out.setZero();
                    for (unsigned int j=0;j<N;j++) {
                        double sum_weight = 0;
                        bool found_one = false;
                        for (unsigned int i=0;i<K;i++) {
                            if (indices(i,j)>=0) {
                                found_one = true;
                                double weight = exp(-dist2(i,j)/weight_sigma);
                                sum_weight += weight;
                                out.block(0,j,OutputDim,1) += 
                                    eigenized_output->block(0,indices(i,j),OutputDim,1)*weight;
                            }
                        }
                        if (sum_weight>1e-10) {
                            out.block(0,j,OutputDim,1) /= sum_weight;
                        } else if (found_one) {
                            out.block(0,j,OutputDim,1) = eigenized_output->block(0,indices(0,j),OutputDim,1);
                        }
                    }
#if 0
                    for (unsigned int j=0;j<N;j++) {
                        for (unsigned int i=0;i<OutputDim;i++) {
                            assert(!isnan(out(i,j)));
                        }
                    }
#endif
                    return out;
                }

                // Input is a special case of Output
                Output operator()(const Input & I) const {
                    return evalAt(I);
                }

                void operator+=(double x) {
                    assert_is_compiled();
                    *eigenized_output += x;
                }

                void operator-=(double x) {
                    assert_is_compiled();
                    *eigenized_output += x;
                }

                void operator+=(const Output & C) {
                    assert_is_compiled();
                    *eigenized_output += C * Eigen::Matrix<T,1,Eigen::Dynamic>::Ones(eigenized_output->cols());
                }

                void operator-=(const Output & C) {
                    assert_is_compiled();
                    *eigenized_output += C * Eigen::Matrix<T,1,Eigen::Dynamic>::Ones(eigenized_output->cols());
                }

                void operator*=(double x) {
                    assert_is_compiled();
                    *eigenized_output *= x;
                }

                void operator/=(double x) {
                    assert_is_compiled();
                    *eigenized_output /= x;
                }

                Function operator+(double x) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output 
                            + Eigen::Matrix<T,OutputDim,Eigen::Dynamic>::Ones(OutputDim,eigenized_output->cols()) * x);
                }
                Function operator-(double x) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output 
                            - Eigen::Matrix<T,OutputDim,Eigen::Dynamic>::Ones(OutputDim,eigenized_output->cols()) * x);
                }

                Function operator*(double x) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output * x);
                }
                Function operator/(double x) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output / x);
                }


                Function operator+(const Output & C) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output 
                            + C * Eigen::Matrix<T,1,Eigen::Dynamic>::Ones(eigenized_output->cols()));
                }

                Function operator-(const Output & C) const {
                    assert_is_compiled();
                    return Function(*eigenized_input,*eigenized_output 
                            - C * Eigen::Matrix<T,1,Eigen::Dynamic>::Ones(eigenized_output->cols()));
                }

                Function operator-() const {
                    assert_is_compiled();
                    return Function(*eigenized_input,-*eigenized_output);
                }
                
                Function operator+() const {
                    assert_is_compiled();
                    return Function(*this);
                }
                
                // No const, because it may be assert_is_compiledd at execution time
                Function operator+(Function & C) const {
                    C.assert_is_compiled(); this->assert_is_compiled();
                    InputMatrix I = mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function(I,this->evalAt(I) + C.evalAt(I));
                }
                Function operator-(Function & C) const {
                    C.assert_is_compiled(); this->assert_is_compiled();
                    InputMatrix I = mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function(I,this->evalAt(I) - C.evalAt(I));
                }

                template <int NewDim>
                    Function<T,InputDim,NewDim> map(Eigen::Matrix<T, NewDim, 1> (*f)(const Output &)) const {
                        assert_is_compiled(); // --> not const
                        Eigen::Matrix<T, NewDim, Eigen::Dynamic> O(NewDim,eigenized_input->cols());
                        for (size_t i=0;i<eigenized_input->cols();i++) {
                            O.block(0,i,NewDim,1) = f(eigenized_output->block(0,i,OutputDim,1));
                        }
                        return Function<T,InputDim,NewDim>(*(this->eigenized_input),O);
                }
                Function<T,InputDim,1> map(double (*f)(const Output &)) const {
                        assert_is_compiled(); // --> not const
                        Eigen::Matrix<T, 1, Eigen::Dynamic> O(eigenized_input->cols());
                        for (size_t i=0;i<eigenized_input->cols();i++) {
                            O(0,i) = f(eigenized_output->block(0,i,OutputDim,1));
                        }
                        return Function<T,InputDim,1>(*(this->eigenized_input),O);
                }

                // This should use IOstream...
                void print(const std::string & filename) const {
                    FILE * fp = fopen(filename.c_str(),"w");
                    assert(fp);
                    assert_is_compiled();
                    for (size_t i=0;i<eigenized_input->cols();i++) {
                        for (size_t j=0;j<InputDim;j++) {
                            fprintf(fp,"%e ", (*eigenized_input)(j,i));
                        }
                        for (size_t j=0;j<OutputDim;j++) {
                            fprintf(fp,"%e ", (*eigenized_output)(j,i));
                        }
                        fprintf(fp,"\n");
                    }
                    fclose(fp);
                }

        };

    template <typename T> 
        class Function21 : public Function<T,2,1> {
            protected:
                typedef Function<T,2,1> Parent;
            public: 
                Function21() :  Parent() {}
                Function21(const Function21 & F) : Parent(F) {}
                Function21(const Function<T,2,1> & F) : Parent(F) {}
                Function21(const typename Parent::InputMatrix & I,
                        const typename Parent::OutputMatrix & O) : Parent(I,O) {}

                double operator()(double x, double y) const {
                    typename Parent::Input I; I << x, y;
                    return evalAt(I)(0,0);
                }
                void set(double x, double y, double z) {
                    typename Parent::Input I; I << x,y;
                    typename Parent::Output O; O << z;
                    this->uncompiled_data.push_back(typename Parent::Record(I,O));
                }
                Function21 map(double (*f)(double)) const {
                    this->assert_is_compiled(); // --> not const
                    Eigen::Matrix<T, 1, Eigen::Dynamic> O(this->eigenized_input->cols());
                    for (size_t i=0;i<this->eigenized_input->cols();i++) {
                        O(0,i) = f((*(this->eigenized_output))(0,i));
                    }
                    return Function21(*(this->eigenized_input),O);
                }
                Function21 operator*(const Function21 & C) const {
                    C.assert_is_compiled();
                    this->assert_is_compiled();
                    typename Parent::InputMatrix I = this->mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function21(I,this->evalAt(I).array() * C.evalAt(I).array());
                }
                Function21 operator/(const Function21 & C) const {
                    C.assert_is_compiled();
                    this->assert_is_compiled();
                    typename Parent::InputMatrix I = this->mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function21(I,this->evalAt(I).array() / C.evalAt(I).array());
                }
        };

    template <typename T> 
        class Function31 : public Function<T,3,1> {
            protected:
                typedef Function<T,3,1> Parent;
            public: 
                Function31() :  Parent() {}
                Function31(const Function31 & F) : Parent(F) {}
                Function31(const Function<T,3,1> & F) : Parent(F) {}
                Function31(const typename Parent::InputMatrix & I,
                        const typename Parent::OutputMatrix & O) : Parent(I,O) {}

                double operator()(double x, double y, double z) const {
                    typename Parent::Input I; I << x, y, z;
                    return evalAt(I)(0,0);
                }
                void set(double x, double y, double z, double w) {
                    typename Parent::Input I; I << x,y,z;
                    typename Parent::Output O; O << w;
                    this->uncompiled_data.push_back(Record(I,O));
                }
                Function31 map(double (*f)(double)) const {
                        this->assert_is_compiled(); // --> not const
                        Eigen::Matrix<T, 1, Eigen::Dynamic> O(this->eigenized_input->cols());
                        for (size_t i=0;i<this->eigenized_input->cols();i++) {
                            O(0,i) = f((*this->eigenized_output)(0,i));
                        }
                        return Function31(*(this->eigenized_input),O);
                }
                Function31 operator*(const Function31 & C) const {
                    C.assert_is_compiled();
                    this->assert_is_compiled();
                    typename Parent::InputMatrix I = this->mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function31(I,this->evalAt(I).array() * C.evalAt(I).array());
                }
                Function31 operator/(const Function31 & C) const {
                    C.assert_is_compiled();
                    this->assert_is_compiled();
                    typename Parent::InputMatrix I = this->mergeSupport(*(this->eigenized_input),*C.eigenized_input);
                    return Function31(I,this->evalAt(I).array() / C.evalAt(I).array());
                }
        };

    template <typename T, int OutputDim> 
        class Function2 : public Function<T,2,OutputDim> {
            protected:
                typedef Function<T,2,OutputDim> Parent;
            public: 
                Function2() :  Parent() {}
                Function2(const Function2 & F) : Parent(F) {}
                Function2(const Function<T,2,OutputDim> & F) : Parent(F) {}
                Function2(const typename Parent::InputMatrix & I,
                        const typename Parent::OutputMatrix & O) : Parent(I,O) {}

                typename Parent::Output operator()(double x, double y) const {
                    typename Parent::Input I; I << x, y;
                    return evalAt(I);
                }

                void set(double x, double y, const typename Parent::Output & O) {
                    typename Parent::Input I; I << x,y;
                    this->uncompiled_data.push_back(Record(I,O));
                }
        };

    template <typename T, int OutputDim> 
        class Function3 : public Function<T,3,OutputDim> {
            protected:
                typedef Function<T,3,OutputDim> Parent;
            public: 
                Function3() :  Parent() {}
                Function3(const Function<T,3,OutputDim> & F) : Parent(F) {}
                Function3(const Function3 & F) : Parent(F) {}
                Function3(const typename Parent::InputMatrix & I,
                        const typename Parent::OutputMatrix & O) : Parent(I,O) {}

                typename Parent::Output operator()(double x, double y, double z) const {
                    typename Parent::Input I; I << x, y, z;
                    return evalAt(I);
                }
                void set(double x, double y, double z, const typename Parent::Output & O) {
                    typename Parent::Input I; I << x,y,z;
                    this->uncompiled_data.push_back(Record(I,O));
                }
        };


    typedef Function21<float> Function21f;
    typedef Function2<float, 2> Function22f;
    typedef Function2<float, 3> Function23f;
    typedef Function21<double> Function21d;
    typedef Function2<double, 2> Function22d;
    typedef Function2<double, 3> Function23d;

    typedef Function31<float> Function31f;
    typedef Function3<float, 2> Function32f;
    typedef Function3<float, 3> Function33f;
    typedef Function31<double> Function31d;
    typedef Function3<double, 2> Function32d;
    typedef Function3<double, 3> Function33d;

};

#endif // NABO_FUNCTION_H
