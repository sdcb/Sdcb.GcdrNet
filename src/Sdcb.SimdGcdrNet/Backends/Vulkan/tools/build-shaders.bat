@echo off
rem Dev-time only: compile the GCDR-specific .comp GLSL -> .spv next to each source (committed).
rem Every shader is built twice: name.spv (fp32 storage) and name_h.spv (-DF16, see Shaders/et.glsl).
setlocal
set "GLSLC=C:\VulkanSDK\1.4.357.0\Bin\glslc.exe"
if not exist "%GLSLC%" for %%i in (glslc.exe) do set "GLSLC=%%~$PATH:i"
if not exist "%GLSLC%" (
    echo glslc not found - install Vulkan SDK or put glslc.exe on PATH
    exit /b 1
)
set "DIR=%~dp0..\Shaders"
set "O=-O --target-env=vulkan1.1"
for %%f in ("%DIR%\*.comp") do (
    echo glslc %%~nxf
    "%GLSLC%" %O% "%%f" -o "%%~dpnf.spv" || exit /b 1
    "%GLSLC%" %O% -DF16 "%%f" -o "%%~dpnf_h.spv" || exit /b 1
)
"%GLSLC%" %O% -DF16 -DNP=2u "%DIR%\uphead_sc.comp" -o "%DIR%\uphead_sc2_h.spv" || exit /b 1
exit /b 0
