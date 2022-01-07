HEIF_LIB=/work/src/heic2hevc/nokiatech-heif

CXXFLAGS = -O2 -std=c++11 -W -Wall
CXXFLAGS += -I$(HEIF_LIB)/srcs/api/common -I$(HEIF_LIB)/srcs/api/reader
LDFLAGS = -L$(HEIF_LIB)/build/lib -lheif_static

all: clean heic2hevc

heic2hevc: heic2hevc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf heic2hevc

.PHONY: clean
