import numpy as np
# 双 EMA 对单位阶跃的分离峰值：div(t) = e^{-t/6} - e^{-t/0.7}
tau_f, tau_s, dt = 0.7, 6.0, 0.01
t = np.arange(0, 30, dt)
div = np.exp(-t/tau_s) - np.exp(-t/tau_f)
print(f"tau_f={tau_f} tau_s={tau_s}: 峰值 div/Δ = {div.max():.4f} @ t={t[div.argmax()]:.2f}s")
print(f"=> 名义 kStepRel=0.18 对应的最小可识别台阶 = 0.18/{div.max():.4f} = {0.18/div.max():.4f} × 当前电平")
print(f"   本数据 240.7s 台阶: Δ=+5254, 电平≈20500 -> r={5254/20500:.3f} (峰值分离 {div.max()*5254:.0f} ADC, 阈值 0.18*slow≈{0.18*20500:.0f})")
print(f"   本数据 202.6s 台阶: Δ=-3919, 电平≈26100 -> r={3919/26100:.3f}")
