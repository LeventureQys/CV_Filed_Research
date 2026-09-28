@echo off
rem Build the v3.1 "baseline keeping" sandbox variants (diagnostic only, not product).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set S=%ROOT%\temp\v4.1flash\plan\v3.1\scripts
set S30=%ROOT%\temp\v4.1flash\plan\v3.0\scripts
set VROOT=%ROOT%\temp\v4.1flash\plan\v3.1\results\v31_variants
if not exist "%VROOT%\build" mkdir "%VROOT%\build"
cd /d "%VROOT%\build"
for %%V in (cur anchor_base anchor_base_nocap anchor_obs) do (
  echo --- %%V ---
  cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%VROOT%\%%V" /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v31_%%V.exe "%S30%\v30_runner.cpp" "%VROOT%\%%V\domain\drift_v6\drift_v6_compensator.cpp"
  if errorlevel 1 exit /b 1
)
echo BUILD OK
