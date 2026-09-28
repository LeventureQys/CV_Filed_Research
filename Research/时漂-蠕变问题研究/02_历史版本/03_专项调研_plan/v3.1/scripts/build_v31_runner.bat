@echo off
rem Build the plan-v3.1 A/B runners:
rem   v30_runner.exe        - PRODUCT source src/domain/drift_v6 (AFTER the fix)
rem   v31_runner_prefix.exe - scripts/prefix/... (BEFORE the fix, counterfactual)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=D:\workshop\Processing\multi-device-cascade-host-cpp
set S=%ROOT%\temp\v4.1flash\plan\v3.1\scripts
set S30=%ROOT%\temp\v4.1flash\plan\v3.0\scripts
cd /d "%S%\build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%ROOT%\src" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v30_runner.exe "%S30%\v30_runner.cpp" "%ROOT%\src\domain\drift_v6\drift_v6_compensator.cpp"
if errorlevel 1 exit /b 1
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "%S%\prefix" /I "%ROOT%\thirdparty\eigen-5.0.0" /Fe:v31_runner_prefix.exe "%S30%\v30_runner.cpp" "%S%\prefix\domain\drift_v6\drift_v6_compensator.cpp"
if errorlevel 1 exit /b 1
echo BUILD OK
