@echo off
SET REPOSITORY_ROOT=%~dp0
cd %REPOSITORY_ROOT%MaterialX
cmake -S . -Bbuild -DMATERIALX_BUILD_PYTHON=OFF -DMATERIALX_BUILD_RENDER=OFF -DMATERIALX_BUILD_TESTS=OFF -DCMAKE_DEBUG_POSTFIX=d  -DMATERIALX_BUILD_GEN_OSL=OFF -DMATERIALX_BUILD_GEN_MDL=OFF -DMATERIALX_BUILD_CONTRIB=OFF -DMATERIALX_BUILD_VIEWER=OFF
cmake --build build --target install --config Debug
cmake --build build --target install --config Release
cd %REPOSITORY_ROOT%