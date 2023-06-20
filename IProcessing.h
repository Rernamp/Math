#pragma once

namespace UDA {
    class IProcessing {
    public:
        static constexpr std::size_t maxNumberElements = 4;
        virtual void process(ITransmitter* transmiter, StaticVector<MapDataContainerToFilter, maxNumberElements>& dataContainers, const std::size_t& processSize) = 0;        
        virtual void start() = 0;
        bool stopRequest = false;
    };
}