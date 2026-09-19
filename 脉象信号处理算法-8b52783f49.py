import numpy as np
from scipy import signal
import datetime  # 导入标准时间模块
import json

class PulseSignalProcessor:
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.filter_order = 32
        self.mu = 0.1  # LMS算法步长
        
    def _lms_filter(self, x, d):
        """实现LMS自适应滤波算法"""
        n = len(x)
        w = np.zeros(self.filter_order)  # 滤波器权重
        y = np.zeros(n)  # 输出信号
        e = np.zeros(n)  # 误差信号
        
        for i in range(self.filter_order, n):
            x_window = x[i-self.filter_order:i][::-1]  # 输入窗口（反转以匹配卷积顺序）
            y[i] = np.dot(w, x_window)
            e[i] = d[i] - y[i]
            w += self.mu * e[i] * x_window  # 更新权重
        
        return y
    
    def load_raw_data(self, file_path):
        """加载原始脉象信号数据"""
        with open(file_path, 'r') as f:
            return np.array(json.load(f)['pulse_data'])
    
    def preprocess(self, raw_signal):
        """信号预处理：去噪与基线校正"""
        # 生成参考噪声（使用原始信号的延迟版本）
        noise_reference = np.roll(raw_signal, int(0.1 * self.sample_rate))
        
        # 应用LMS自适应滤波
        filtered = self._lms_filter(raw_signal, noise_reference)
        
        # 带通滤波（0.5-20Hz）
        b, a = signal.butter(4, [0.5, 20], btype='bandpass', fs=self.sample_rate)
        filtered = signal.filtfilt(b, a, filtered)
        
        # 基线校正
        baseline = signal.savgol_filter(filtered, window_length=101, polyorder=3)
        corrected = filtered - baseline
        
        return corrected
    
    def extract_time_domain_features(self, pulse_signal):
        """提取时域特征"""
        t = np.arange(len(pulse_signal)) / self.sample_rate
        
        # 主波特征
        h1 = np.max(pulse_signal)
        t1 = t[np.argmax(pulse_signal)]
        
        # 重搏波特征
        peak_indices = signal.argrelmax(np.abs(pulse_signal), order=30)[0]
        valid_peaks = peak_indices[pulse_signal[peak_indices] > 0.3 * h1]
        
        h3 = pulse_signal[valid_peaks[1]] if len(valid_peaks) > 1 else 0
        t3 = t[valid_peaks[1]] if len(valid_peaks) > 1 else 0
        
        # 降中峡特征
        trough_indices = signal.argrelmin(np.abs(pulse_signal), order=20)[0]
        post_max_troughs = trough_indices[trough_indices > np.argmax(pulse_signal)]
        h4 = np.min(pulse_signal[post_max_troughs]) if len(post_max_troughs) > 0 else 0
        
        return {
            'h1': h1, 't1': t1,
            'h3': h3, 't3': t3,
            'h4': h4,
            'pulse_rate': self._calculate_pulse_rate(pulse_signal, t)
        }
    
    def _calculate_pulse_rate(self, pulse_signal, t):
        """计算脉率（次/分钟）"""
        peaks, _ = signal.find_peaks(pulse_signal, height=0.3*np.max(pulse_signal), distance=0.3*self.sample_rate)
        if len(peaks) < 2:
            return 0
        intervals = np.diff(t[peaks])
        return 60 / np.mean(intervals)
    
    def extract_freq_domain_features(self, pulse_signal):
        """提取频域特征"""
        freqs, psd = signal.welch(pulse_signal, fs=self.sample_rate, nperseg=512)
        dominant_freq = freqs[np.argmax(psd)]
        
        # 功率谱能比（PSR）
        low_freq_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 3)])
        total_power = np.sum(psd)
        psr = low_freq_power / total_power if total_power > 0 else 0
        
        return {
            'dominant_freq': dominant_freq,
            'psr': psr,
            'spectral_entropy': self._calculate_spectral_entropy(freqs, psd)
        }
    
    def _calculate_spectral_entropy(self, freqs, psd):
        """计算频谱熵"""
        psd_normalized = psd / np.sum(psd)
        return -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
    
    def analyze(self, raw_signal):
        """完整分析流程"""
        processed = self.preprocess(raw_signal)
        time_features = self.extract_time_domain_features(processed)
        freq_features = self.extract_freq_domain_features(processed)
        
        return {
            'time_domain': time_features,
            'frequency_domain': freq_features,
            'timestamp': datetime.datetime.now().isoformat()  # 使用标准库生成时间戳
        }

# 执行示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = PulseSignalProcessor()
    
    # 生成模拟脉象信号（实际应用中从传感器读取）
    t = np.linspace(0, 10, 2500)  # 10秒数据
    raw_signal = 0.5*np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2.5*t) + 0.1*np.random.randn(len(t))
    
    # 执行分析
    result = processor.analyze(raw_signal)
    
    # 保存结果
    with open('脉象特征分析结果.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("脉象特征提取完成，结果已保存至'脉象特征分析结果.json'")
    print(f"脉率: {result['time_domain']['pulse_rate']:.1f} 次/分钟")
    print(f"主频率: {result['frequency_domain']['dominant_freq']:.2f} Hz")