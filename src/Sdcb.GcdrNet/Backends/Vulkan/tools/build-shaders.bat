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
    if not "%%~nxf"=="conv_tile.comp" if not "%%~nxf"=="conv_cm.comp" if not "%%~nxf"=="sgprobe.comp" if not "%%~nxf"=="conv_dot.comp" if not "%%~nxf"=="gemm_dot.comp" (
        echo glslc %%~nxf
        "%GLSLC%" %O% "%%f" -o "%%~dpnf.spv" || exit /b 1
        "%GLSLC%" %O% -DF16 "%%f" -o "%%~dpnf_h.spv" || exit /b 1
    )
)
rem conv_tile specialization variants (constant kernel/stride -> fast index math)
call :conv "" ""   || exit /b 1
call :conv -DF16 _h || exit /b 1
rem tensor-core conv (fp16 storage only): BN=128 / 64 / 32 output-channel tiles
set "CM=%DIR%\conv_cm.comp"
"%GLSLC%" %O% "%CM%" -o "%DIR%\conv_cm_h.spv" || exit /b 1
"%GLSLC%" %O% -DSGX=2u -DSGY=2u -DTSM=4u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_n64_h.spv" || exit /b 1
"%GLSLC%" %O% -DSGX=1u -DSGY=4u -DTSM=2u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_n32_h.spv" || exit /b 1
rem any-cin stems (k elements gathered one by one)
"%GLSLC%" %O% -DSCALARK -DSGX=1u -DSGY=4u -DTSM=2u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_n32s_h.spv" || exit /b 1
rem LITE: the same four for wave64-capable / 32 KB shared-memory parts (subgroup pinned to 32);
rem more, narrower (64x16) subgroup strips per tile measured fastest on the Radeon 880M
"%GLSLC%" %O% -DLITE -DSGX=8u -DSGY=2u -DTSM=4u -DTSN=1u -DNTHR=512 "%CM%" -o "%DIR%\conv_cm_l_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSGX=4u -DSGY=2u -DTSM=4u -DTSN=1u -DNTHR=256 "%CM%" -o "%DIR%\conv_cm_l_n64_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSGX=2u -DSGY=2u -DTSM=4u -DTSN=1u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_l_n32_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSCALARK -DSGX=1u -DSGY=4u -DTSM=2u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_l_n32s_h.spv" || exit /b 1
rem SG16: 8x16x16 fragments on 16-lane subgroups (Intel Xe2 XMX), LITE staging, BM = 128;
rem a 16x32 strip per subgroup, eight stacked along M, measured fastest on the Arc B580
"%GLSLC%" %O% -DLITE -DSG16 -DSGX=4u -DSGY=8u -DTSM=2u -DTSN=2u -DNTHR=512 "%CM%" -o "%DIR%\conv_cm_sg16_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSG16 -DSGX=2u -DSGY=8u -DTSM=2u -DTSN=2u -DNTHR=256 "%CM%" -o "%DIR%\conv_cm_sg16_n64_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSG16 -DSGX=1u -DSGY=8u -DTSM=2u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_sg16_n32_h.spv" || exit /b 1
"%GLSLC%" %O% -DLITE -DSG16 -DSCALARK -DSGX=1u -DSGY=8u -DTSM=2u -DTSN=2u -DNTHR=128 "%CM%" -o "%DIR%\conv_cm_sg16_n32s_h.spv" || exit /b 1
rem wide-wave no-coopmat GPUs (Adreno): register-tiled direct kernels, fp16 storage only
set "CD=%DIR%\conv_dot.comp"
"%GLSLC%" %O% -DSW=1u -DTM=4 -DTN=2 "%CD%" -o "%DIR%\conv_dot_s1_4x2_h.spv" || exit /b 1
"%GLSLC%" %O% -DSW=1u -DTM=4 -DTN=1 "%CD%" -o "%DIR%\conv_dot_s1_4x1_h.spv" || exit /b 1
"%GLSLC%" %O% -DSW=2u -DTM=2 -DTN=2 "%CD%" -o "%DIR%\conv_dot_s2_2x2_h.spv" || exit /b 1
"%GLSLC%" %O% -DSW=2u -DTM=2 -DTN=1 "%CD%" -o "%DIR%\conv_dot_s2_2x1_h.spv" || exit /b 1
"%GLSLC%" %O% "%DIR%\gemm_dot.comp" -o "%DIR%\gemm_dot_h.spv" || exit /b 1
"%GLSLC%" %O% -DF16 -DPIXELS_INNER "%DIR%\convpool.comp" -o "%DIR%\convpool_pi_h.spv" || exit /b 1
"%GLSLC%" %O% -DF16 -DNP=2u "%DIR%\uphead_sc.comp" -o "%DIR%\uphead_sc2_h.spv" || exit /b 1
rem subgroup-size probe for pinned (requiredSubgroupSize = 32 / 16) pipelines
"%GLSLC%" %O% "%DIR%\sgprobe.comp" -o "%DIR%\sgprobe.spv" || exit /b 1
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
