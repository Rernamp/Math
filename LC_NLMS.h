#include <arm_math.h>


#include <W5500Connection.h>
#include <DataContainer.h>
#include <Math/IProcessing.h>
#include <Math/IProcessing.h>
#include <StaticVector.h>

#include <array>
#include <limits>

namespace UDA {

    class LC_NLMS : public IProcessing {
        static constexpr float mu = 0.1f;
    public: 
        template<typename ValueType = float32_t>
        struct BufferPointer {
        	ValueType * data;
            std::size_t size;
        };
        void initFilter(std::size_t numberElements, std::size_t filterOrders, std::size_t sizeProcessing = numberSizeProcessing) {
            _numberElements = numberElements;
            _filterOrders = filterOrders;
            _weightingCoefficient.fill(0.0f);
            _linearConstr.fill(0.0f);

            _linearConstr[filterOrders - 1] = 1.0f;
            for (std::size_t i = 0; i < _numberElements; i++) {
                _weightingCoefficientPountes[i].data = &_weightingCoefficient[i * _filterOrders];
                _weightingCoefficientPountes[i].size = _filterOrders;
                _weightingCoefficientPountes[i].data[_filterOrders - 1] = 0.5f;
                _filterStateBufferPointer[i].data = &_filterStateBuffer[i * (sizeProcessing + _filterOrders - 1)];
                arm_lms_norm_init_f32(&_filtersInstance[i], static_cast<uint16_t>(_filterOrders), _weightingCoefficientPountes[i].data, _filterStateBufferPointer[i].data, mu, sizeProcessing);
            }
        }
        void process(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers, const std::size_t& processSize) override {
            for (std::size_t j = 0; j < processSize; j++) {
                for (std::size_t i = 0; i < dataContainers.size(); i++) {
                    _filtersInputBufferPointer[i].data = dataContainers[i].container.data;
                    _filtersInputBufferPointer[i].size = 1;
                    ++dataContainers[i].container.data;
                }
                multichannel_lms_norm_FROST_f32(&_filtersInstance[0], &_filtersInputBufferPointer[0],
                &_linearConstr[0], &_outputBuffer[0], _filtersInputBufferPointer[0].size);

                transmiter->append(reinterpret_cast<uint8_t*>(&_outputBuffer[0]), 1 * sizeof(float32_t));
            }
            
        }
        void start() override {
        }

    private:
        static constexpr std::size_t maxNumberElements = 4;
        static constexpr std::size_t maxFilterOrders = 32;
        static constexpr std::size_t maxNumberCoefficient = maxNumberElements * maxFilterOrders;
        
        std::array<arm_lms_norm_instance_f32, maxNumberElements> _filtersInstance {};

        std::array<float32_t, maxNumberCoefficient> _weightingCoefficient {};
        std::array<BufferPointer<float32_t>, maxNumberElements> _weightingCoefficientPountes {};

        

        static constexpr std::size_t numberSizeProcessing = 128;
        static constexpr std::size_t numberForOneFilter = numberSizeProcessing + maxFilterOrders - 1;
        static constexpr std::size_t maxSizeStateBuffer = (numberForOneFilter) * maxNumberElements ;

        
        std::array<float32_t, maxSizeStateBuffer> _filterStateBuffer {};
        std::array<BufferPointer<float32_t>, maxNumberElements> _filterStateBufferPointer {};
        std::array<BufferPointer<int16_t>, maxNumberElements> _filtersInputBufferPointer {};

        std::array<float32_t, numberSizeProcessing> _outputBuffer {};

        std::array<float32_t, numberSizeProcessing> _linearConstr {};
        
        std::size_t _numberElements = maxNumberElements;
        std::size_t _filterOrders = maxFilterOrders;


    struct FilterDetailt {
        float32_t *pState;
        float32_t *pCoeffs;
        float32_t *pStateCurnt;
        float32_t *px, *pb, *pb_s2;
        float32_t energy;    
        float32_t sum;       
        float32_t x0, in;   
    };

    std::array<FilterDetailt, maxNumberElements> _filtersDetailt {};

    void multichannel_lms_norm_FROST_f32(
    arm_lms_norm_instance_f32 * filterInstance,
    BufferPointer<int16_t> * inputPointer,
    float32_t * pCvec,                    // linear constrain
    float32_t * pOut,
    uint32_t blockSize) {
        {
            arm_lms_norm_instance_f32* tempInstance = filterInstance;
            for (std::size_t i = 0; i < _numberElements; i++) {
                _filtersDetailt[i].pState = tempInstance->pState;
                _filtersDetailt[i].pCoeffs = tempInstance->pCoeffs;
                _filtersDetailt[i].pStateCurnt = &(tempInstance->pState[(tempInstance->numTaps - 1U)]);
                _filtersDetailt[i].energy = tempInstance->energy;
                _filtersDetailt[i].x0 = tempInstance->x0;
                tempInstance++;
            }
        }
        // ......................... common..............
        float32_t *pc;                                      //Temporary pointers for linear constrain
        float32_t w, e;                                      //weight factor, error
        float32_t v;                                      //
        float32_t mu = filterInstance->mu;                          /* Adaptive factor */
        uint32_t  numTaps = filterInstance->numTaps;                 /* Number of filter coefficients in the filter */
        uint32_t  tapCnt, blkCnt; /* Loop counters */

        float32_t totalEnergy = 0.0f;
        
        /* Initializations of error,  difference, Coefficient update */
        e = 0.0f;
        w = 0.0f;
        v = 0.0f;            
        /* Loop over blockSize number of values */
        blkCnt = blockSize;

        FilterDetailt* filterDetailt = &_filtersDetailt[0];
        const FilterDetailt* stopFilterDetailt = &_filtersDetailt[_numberElements - 1];
        BufferPointer<int16_t>* _tempInputPointers = inputPointer;
        /* Run the below code for Cortex-M4 and Cortex-M3 */

            while (blkCnt > 0U)
            {
                filterDetailt = &_filtersDetailt[0];
                _tempInputPointers = inputPointer;

                *pOut = 0.0f;

                totalEnergy = 0.0f;

                while(filterDetailt <= stopFilterDetailt) {
                    
                    static constexpr float32_t scaleValue = 1.0f;
                    // // *filterDetailt->pStateCurnt = static_cast<float32_t>(*_tempInputPointers->data) / static_cast<float32_t>(std::numeric_limits<int16_t>::max()) * scaleValue;
                    *filterDetailt->pStateCurnt = static_cast<float32_t>(*_tempInputPointers->data);
                    filterDetailt->px = filterDetailt->pState;
                    filterDetailt->pb = filterDetailt->pCoeffs;
                    
                    // filterDetailt->in = static_cast<float32_t>(*_tempInputPointers->data) / static_cast<float32_t>(std::numeric_limits<int16_t>::max()) * scaleValue;
                    filterDetailt->in = static_cast<float32_t>(*_tempInputPointers->data);
                    ++_tempInputPointers->data;
                    // filterDetailt->in = *_tempInputPointers->data++;
                    filterDetailt->energy -= filterDetailt->x0 * filterDetailt->x0;
                    filterDetailt->energy += filterDetailt->in * filterDetailt->in;

                    filterDetailt->sum = 0.0f;

                    tapCnt = numTaps >> 2;

                    while(tapCnt > 0) {
                        filterDetailt->sum += (*filterDetailt->px++) * (*filterDetailt->pb++);
                        filterDetailt->sum += (*filterDetailt->px++) * (*filterDetailt->pb++);
                        filterDetailt->sum += (*filterDetailt->px++) * (*filterDetailt->pb++);
                        filterDetailt->sum += (*filterDetailt->px++) * (*filterDetailt->pb++);
                        tapCnt--;
                    }      

                    tapCnt = numTaps % 0x4U;  

                    while (tapCnt > 0U) {
                        filterDetailt->sum += (*filterDetailt->px++) * (*filterDetailt->pb++);
                        tapCnt--;
                    }  

                    totalEnergy += filterDetailt->energy;

                    *pOut += filterDetailt->sum; 
                    ++_tempInputPointers;
                    ++filterDetailt;
                }

                // start next iteration
                
                /* Compute and store error */
        //        d = (float32_t) (*pRef++);
        //        e = d - sum;
                e = - (*pOut);

                ++pOut; 

                /* Calculation of Weighting factor for updating filter coefficients */
                /* epsilon value 0.000000119209289f */
                static constexpr float32_t minimalValue = 0.000000119209289f;
                w = (e * mu) / (totalEnergy + minimalValue);

                /* Initialize pState pointer */                
                filterDetailt = &_filtersDetailt[0];

                while(filterDetailt <= stopFilterDetailt) {
                    filterDetailt->px = filterDetailt->pState;
                    filterDetailt->pb = filterDetailt->pCoeffs;

                    tapCnt = numTaps >> 2;

                    /* Update filter coefficients */
                    while (tapCnt > 0U) {
                        *filterDetailt->pb += w * (*filterDetailt->px++);
                        ++filterDetailt->pb;

                        *filterDetailt->pb += w * (*filterDetailt->px++);
                        ++filterDetailt->pb;

                        *filterDetailt->pb += w * (*filterDetailt->px++);
                        ++filterDetailt->pb;

                        *filterDetailt->pb += w * (*filterDetailt->px++);
                        ++filterDetailt->pb;

                        --tapCnt;
                    }

                    tapCnt = numTaps % 0x4U;

                    while (tapCnt > 0U) {
                        *filterDetailt->pb += w * (*filterDetailt->px++);
                        ++filterDetailt->pb;
                        --tapCnt;
                    }


                    ++filterDetailt;
                }

                filterDetailt = &_filtersDetailt[0];

        //....................................2nd stage - my .......................................//
            
                pc = (pCvec);
                /* Initialize coeff pointer */

                while(filterDetailt <= stopFilterDetailt) {
                    filterDetailt->pb_s2 = filterDetailt->pCoeffs;
                    ++filterDetailt;
                }
                filterDetailt = &_filtersDetailt[0];
            
                tapCnt = numTaps >> 2;

                while(tapCnt > 0) {
                    //I
                    v = *pc;
                    while(filterDetailt <= stopFilterDetailt) {
                        v -= *filterDetailt->pb_s2;
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    v /= 2.0f;

                    while(filterDetailt <= stopFilterDetailt) {
                        *filterDetailt->pb_s2 += v;
                        ++filterDetailt->pb_s2;                        
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    ++pc;

                    //II
                    v = *pc;
                    while(filterDetailt <= stopFilterDetailt) {
                        v -= *filterDetailt->pb_s2;
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    v /= 2.0f;

                    while(filterDetailt <= stopFilterDetailt) {
                        *filterDetailt->pb_s2 += v;
                        ++filterDetailt->pb_s2;                        
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    ++pc;

                    //III
                    v = *pc;
                    while(filterDetailt <= stopFilterDetailt) {
                        v -= *filterDetailt->pb_s2;
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    v /= 2.0f;

                    while(filterDetailt <= stopFilterDetailt) {
                        *filterDetailt->pb_s2 += v;
                        ++filterDetailt->pb_s2;                        
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    ++pc;

                    //IV
                    v = *pc;
                    while(filterDetailt <= stopFilterDetailt) {
                        v -= *filterDetailt->pb_s2;
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    v /= 2.0f;

                    while(filterDetailt <= stopFilterDetailt) {
                        *filterDetailt->pb_s2 += v;
                        ++filterDetailt->pb_s2;                        
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    ++pc;

                    --tapCnt;
                }
                /* If the filter length is not a multiple of 4, compute the remaining filter taps */
                tapCnt = numTaps % 0x4U;

                while (tapCnt > 0U)
                {
                    /* Perform the multiply-accumulate */
                    v = *pc;
                    while(filterDetailt <= stopFilterDetailt) {
                        v -= *filterDetailt->pb_s2;
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    v /= 2.0f;

                    while(filterDetailt <= stopFilterDetailt) {
                        *filterDetailt->pb_s2 += v;
                        ++filterDetailt->pb_s2;                        
                        ++filterDetailt;
                    }
                    filterDetailt = &_filtersDetailt[0];
                    ++pc;

                    /* Decrement the loop counter */
                    tapCnt--;
                }       
                
                while(filterDetailt <= stopFilterDetailt) {
                    filterDetailt->x0 = *filterDetailt->pState;
                    ++filterDetailt->pState;
                    ++filterDetailt;
                }
                filterDetailt = &_filtersDetailt[0];
                /* Decrement the loop counter */
                blkCnt--;
            }
        //....................................End of block proc.......................................//
            {
                arm_lms_norm_instance_f32* tempInstance = filterInstance;
                while(filterDetailt <= stopFilterDetailt) {
                    tempInstance->x0 = filterDetailt->x0;
                    tempInstance->energy = filterDetailt->energy;
                    /* Processing is complete. Now copy the last numTaps - 1 samples to the
                    start of the state buffer. This prepares the state buffer for the
                    next function call. */
                    /* Points to the start of the pState buffer */
                    filterDetailt->pStateCurnt = tempInstance->pState;
                    ++filterDetailt;
                    ++tempInstance;
                }
                filterDetailt = &_filtersDetailt[0];
            }        

            

            while(filterDetailt <= stopFilterDetailt) {
                tapCnt = (numTaps - 1U) >> 2U;
                while (tapCnt > 0U) {
                    *filterDetailt->pStateCurnt++ = *filterDetailt->pState++;
                    *filterDetailt->pStateCurnt++ = *filterDetailt->pState++;
                    *filterDetailt->pStateCurnt++ = *filterDetailt->pState++;
                    *filterDetailt->pStateCurnt++ = *filterDetailt->pState++;    
                    tapCnt--;                
                }
                    
                /* Calculate remaining number of copies */
                tapCnt = (numTaps - 1U) % 0x4U;

                /* Copy the remaining q31_t data */
                while (tapCnt > 0U)
                {
                    *filterDetailt->pStateCurnt++ = *filterDetailt->pState++;                                        
                    tapCnt--;
                }

                ++filterDetailt;
            }
            filterDetailt = &_filtersDetailt[0];

    }
    };



}
