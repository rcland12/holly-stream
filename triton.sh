#!/bin/bash
./tritonserver/bin/tritonserver \
--model-repository=./triton/ \
--backend-directory=./tritonserver/backends/
