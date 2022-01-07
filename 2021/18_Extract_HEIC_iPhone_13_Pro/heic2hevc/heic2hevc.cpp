/*
 * heic2hevc.cpp -- convert HEIC file to H.265 bitstream(Annex.B)
 * depends https://github.com/nokiatech/heif
 */
#include <iostream>
#include "heifreader.h"

using namespace HEIF;

void printInfo(const char* file);
void extractItemId(const char* srcfile, const char* dstfile);
void example5(const char* srcfile, const char* dstfile);


void extractItemId(const char* srcfile, const char* dstfile)
{
    std::cout << "convert " << srcfile << " to " << dstfile << std::endl;

    auto* reader = Reader::Create();
    // Input file available from https://github.com/nokiatech/heif_conformance
    const char* filename = srcfile;
    if (reader->initialize(filename) != ErrorCode::OK)
    {
        printInfo(filename);
        Reader::Destroy(reader);
        return;
    }

    FileInformation info;
    reader->getFileInformation(info);

    // print 
    auto metaBoxInfo = info.rootMetaBoxInformation;
    for (const auto& itemInformation : metaBoxInfo.itemInformations){
        // char *descriptionName = itemInformation.description.name.elements;
        auto itemId = itemInformation.itemId.get();
        auto descSize = itemInformation.type.value;
        std::cout << "itemId=" << itemId << ", size=" << descSize << std::endl;
        break;
    }

    // Find the item ID
    ImageId itemId;
    reader->getPrimaryItem(itemId);

    uint64_t memoryBufferSize = 1024 * 1024;
    auto* memoryBuffer        = new uint8_t[memoryBufferSize];
    reader->getItemDataWithDecoderParameters(itemId, memoryBuffer, memoryBufferSize);

    // Feed 'data' to decoder and display the cover image...

    delete[] memoryBuffer;

    Reader::Destroy(reader);

}


/// Access and read media track samples, thumbnail track samples and timestamps
void example5(const char* srcfile, const char* dstfile)
{
    auto* reader = Reader::Create();
    Array<uint32_t> itemIds;

    // Input file available from https://github.com/nokiatech/heif_conformance
    const char* filename = srcfile;
    if (reader->initialize(filename) != ErrorCode::OK)
    {
        printInfo(filename);
        Reader::Destroy(reader);
        return;
    }
    FileInformation info;
    reader->getFileInformation(info);

    // Print information for every track read
    for (const auto& trackProperties : info.trackInformation)
    {
        const auto sequenceId = trackProperties.trackId;
        std::cout << "Track ID " << sequenceId.get() << std::endl;  // Context ID corresponds to the track ID

        if (trackProperties.features & TrackFeatureEnum::IsMasterImageSequence)
        {
            std::cout << "This is a master image sequence." << std::endl;
        }

        if (trackProperties.features & TrackFeatureEnum::IsThumbnailImageSequence)
        {
            // Assume there is only one type track reference, so check reference type and master track ID(s) from
            // the first one.
            const auto tref = trackProperties.referenceTrackIds[0];
            std::cout << "Track reference type is '" << tref.type.value << "'" << std::endl;
            std::cout << "This is a thumbnail track for track ID ";
            for (const auto masterTrackId : tref.trackIds)
            {
                std::cout << masterTrackId.get() << std::endl;
            }
        }

        Array<TimestampIDPair> timestamps;
        reader->getItemTimestamps(sequenceId, timestamps);
        std::cout << "Sample timestamps:" << std::endl;
        for (const auto& timestamp : timestamps)
        {
            std::cout << " Timestamp=" << timestamp.timeStamp << "ms, sample ID=" << timestamp.itemId.get() << std::endl;
        }

        for (const auto& sampleProperties : trackProperties.sampleProperties)
        {
            // A sample might have decoding dependencies. The simplest way to handle this is just to always ask and
            // decode all dependencies.
            Array<SequenceImageId> itemsToDecode;
            reader->getDecodeDependencies(sequenceId, sampleProperties.sampleId, itemsToDecode);
            for (auto dependencyId : itemsToDecode)
            {
                uint64_t size    = 1024 * 1024;
                auto* sampleData = new uint8_t[size];
                reader->getItemDataWithDecoderParameters(sequenceId, dependencyId, sampleData, size);

                // Feed data to decoder...

                delete[] sampleData;
            }
            // Store or show the image...
        }
    }

    Reader::Destroy(reader);
}


void printInfo(const char* filename)
{
    std::cout << "Can't find input file: " << filename << ". "
         << "Please download it from https://github.com/nokiatech/heif_conformance "
         << "and place it in same directory with the executable." << std::endl;
}


int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cout
            << "Usage: heic2hevc <input.heic> <output.265>" << std::endl;
        return 0;
    }
    // extractItemId(argv[1], argv[2]);
    example5(argv[1], argv[2]);
    return 0;
}
