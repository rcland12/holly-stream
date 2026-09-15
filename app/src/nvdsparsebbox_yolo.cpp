/*
 * DeepStream (nvinfer) bounding-box parser for YOLO models exported by
 * models/export.py.
 *
 * The model has a single output tensor "output" shaped [N, 6] (batch dimension
 * stripped by nvinfer), one row per candidate:
 *
 *     x1, y1, x2, y2, score, class_id     (network-input pixel coordinates)
 *
 * Rows below the per-class pre-cluster-threshold are dropped here; nvinfer then
 * clusters the survivors (NMS for YOLOv8/YOLO11, nothing for NMS-free YOLO26)
 * and maps the boxes back to frame coordinates, undoing the letterbox.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

static const NvDsInferLayerInfo *findOutputLayer(std::vector<NvDsInferLayerInfo> const &layers)
{
    for (auto const &layer : layers) {
        if (!layer.isInput && layer.layerName && std::strcmp(layer.layerName, "output") == 0)
            return &layer;
    }
    return layers.size() == 1 ? &layers[0] : nullptr;
}

bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    const NvDsInferLayerInfo *layer = findOutputLayer(outputLayersInfo);
    if (!layer) {
        std::cerr << "NvDsInferParseYolo: could not find output layer \"output\"" << std::endl;
        return false;
    }

    const NvDsInferDims &dims = layer->inferDims;
    if (layer->dataType != FLOAT || dims.numDims != 2 || dims.d[1] != 6) {
        std::cerr << "NvDsInferParseYolo: expected a float [N, 6] output, got numDims=" << dims.numDims
                  << " dataType=" << layer->dataType << ". Re-export the model with models/export.py" << std::endl;
        return false;
    }

    const unsigned int numRows = dims.d[0];
    const unsigned int numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float *rows = static_cast<const float *>(layer->buffer);

    objectList.clear();
    for (unsigned int i = 0; i < numRows; ++i) {
        const float *row = rows + i * 6;
        const float score = row[4];
        const int classId = static_cast<int>(std::lround(row[5]));

        if (classId < 0 || static_cast<unsigned int>(classId) >= numClasses)
            continue;
        if (score < detectionParams.perClassPreclusterThreshold[classId])
            continue;

        const float x1 = std::min(std::max(row[0], 0.0f), netW);
        const float y1 = std::min(std::max(row[1], 0.0f), netH);
        const float x2 = std::min(std::max(row[2], 0.0f), netW);
        const float y2 = std::min(std::max(row[3], 0.0f), netH);
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f)
            continue;

        NvDsInferParseObjectInfo object;
        object.classId = static_cast<unsigned int>(classId);
        object.left = x1;
        object.top = y1;
        object.width = x2 - x1;
        object.height = y2 - y1;
        object.detectionConfidence = score;
        objectList.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
