#pragma once

#include <IProcessing.h>

#include "arm_math.h"

#include <array>

namespace UDA {
    class LC_NLMS : public IProcessing {
    public:
        void setFilterOrderAndNUmberElements(std::size_t currentNumberElements, std::size_t currentFilterOrder) {
            _currentFilterOrder = currentFilterOrder;
            _currentNumberElements = currentNumberElements;
            updateFiltersArray();
        }
        void process(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers, const std::size_t& processSize) override {
            for (std::size_t i = 0; i < processSize; i++) {
                do_sample(transmiter, dataContainers);
                for (std::size_t j = 0; j < dataContainers.size(); j++) {
                    ++dataContainers[j].container.data;
                }
            }
        }
        void start() override {

        }    
    private:
        void do_sample(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers) {

        }
        
        void updateFiltersArray() {

        }
        static constexpr std::size_t maxNumberElements = 4;
        static constexpr std::size_t maxFilterOrder = 32;
        static constexpr std::size_t maxFilterElementsSize = maxNumberElements * maxFilterOrder;
        std::array<float32_t, maxFilterElementsSize> _weightingCoefficient {};
        std::array<float32_t, maxFilterElementsSize> _dataPoints {};

        std::size_t _currentNumberElements = maxNumberElements;
        std::size_t _currentFilterOrder = maxFilterOrder;
    };
}