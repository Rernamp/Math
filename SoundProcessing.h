#pragma once

#include <W5500Connection.h>
#include <DataContainer.h>
#include <StaticVector.h>

#include <Math/IProcessing.h>

namespace UDA {


    class AverageProcess : public IProcessing {
    public: 
        void process(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers, const std::size_t& processSize) override {
            int32_t tempData = 0;
            int32_t numberElements = dataContainers.size();
            int16_t data = 0;
            for (std::size_t i = 0; i < processSize; i++) {
            
                for (auto& element : dataContainers) {
                    tempData += *element.container.data;
                    ++element.container.data;
                }
                    
                tempData /= numberElements;
                data = static_cast<int16_t>(tempData);
                transmiter->append(reinterpret_cast<uint8_t*>(&data), sizeof(int16_t));
                tempData = 0;                    
            }
        }
        void start() override {
        }
    };

    class StreamAllSignal : public IProcessing {
    public: 
        void process(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers, const std::size_t& processSize) override {
            for (std::size_t i = 0; i < processSize; i++) {            
                for (auto& element : dataContainers) {
                    while((transmiter->append(reinterpret_cast<uint8_t*>(element.container.data), sizeof(int16_t)) == 0) && (!stopRequest)) {
                    	Eni::Threading::ThisThread::yield();
                    }
                    ++element.container.data;
                }
            }
        }
        void start() override {
        }
    };
}
