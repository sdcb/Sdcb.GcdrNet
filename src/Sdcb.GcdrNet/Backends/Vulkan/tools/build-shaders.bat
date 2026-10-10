@echo off
rem Dev-time only: compile .comp GLSL -> .spv next to each source (committed).
rem Not part of `dotnet build` - SPIR-V is embedded as a resource.
rem Every shader is built twice: name.spv (fp32 storage) and name_h.spv
rem (-DF16: 16-bit storage, fp32 math; see Shaders/et.glsl).
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
    if not "%%~nxf"=="conv_tile.comp" (
        echo glslc %%~nxf
        "%GLSLC%" %O% "%%f" -o "%%~dpnf.spv" || exit /b 1
        "%GLSLC%" %O% -DF16 "%%f" -o "%%~dpnf_h.spv" || exit /b 1
    )
)
rem conv_tile specialization variants (constant kernel/stride -> fast index math)
call :conv "" ""   || exit /b 1
call :conv -DF16 _h || exit /b 1
exit /b 0

:conv
set "CT=%DIR%\conv_tile.comp"
set "F=%~1"
set "S=%~2"
"%GLSLC%" %O% %F% -DKH=3u -DKW=3u -DSH=1u -DSW=1u "%CT%" -o "%DIR%\conv_k33s1%S%.spv" || exit /b 1
"%GLSLC%" %O% %F% -DKH=3u -DKW=3u -DSH=2u -DSW=2u "%CT%" -o "%DIR%\conv_k33s2%S%.spv" || exit /b 1
"%GLSLC%" %O% %F% -DKH=1u -DKW=1u -DSH=1u -DSW=1u "%CT%" -o "%DIR%\conv_k11s1%S%.spv" || exit /b 1
"%GLSLC%" %O% %F% "%CT%" -o "%DIR%\conv_gen%S%.spv" || exit /b 1
exit /b 0
