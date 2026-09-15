DS_DIR ?= /opt/nvidia/deepstream/deepstream
CUDA_DIR ?= /usr/local/cuda-10.2

APP := holly-stream
PARSER := libnvdsinfer_custom_impl_yolo.so

CFLAGS += -O2 -Wall -DPLATFORM_TEGRA -I$(DS_DIR)/sources/includes $(shell pkg-config --cflags gstreamer-1.0)
LIBS += $(shell pkg-config --libs gstreamer-1.0) -L$(DS_DIR)/lib -lnvdsgst_meta -lnvds_meta -Wl,-rpath,$(DS_DIR)/lib

CXXFLAGS += -O2 -Wall -shared -fPIC -std=c++11 -I$(DS_DIR)/sources/includes -I$(CUDA_DIR)/include

all: $(APP) $(PARSER)

$(APP): holly_stream.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(PARSER): nvdsparsebbox_yolo.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(APP) $(PARSER)

.PHONY: all clean
