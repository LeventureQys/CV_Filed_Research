@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "D:\workshop\Processing\multi-device-cascade-host-cpp\temp\v4.1flash\plan\v2.0\scripts\build"
cl /nologo /EHsc /utf-8 /std:c++17 /O2 /I "D:\workshop\Processing\multi-device-cascade-host-cpp\src" /I "D:\workshop\Processing\multi-device-cascade-host-cpp\thirdparty\eigen-5.0.0" /Fe:batch_runner_exp.exe "D:\workshop\Processing\multi-device-cascade-host-cpp\temp\v4.1flash\plan\v2.0\scripts\batch_runner_exp.cpp" "D:\workshop\Processing\multi-device-cascade-host-cpp\src\domain\drift_v6\drift_v6_compensator.cpp"
