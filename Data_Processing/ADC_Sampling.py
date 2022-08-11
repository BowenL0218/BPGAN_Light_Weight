import math
import numpy as np
def ADC_Sampling(audio, target_SNR =100,ADC_bits=8, target_dBSPL=60):
    fclk = 8000
    #######AFE Chain
    mic_out_dbv = -40 - (90 - target_dBSPL)
    target_sig_vrms = 10 ** (mic_out_dbv / 20)
    target_noise_vrms = target_sig_vrms / 10 ** (target_SNR / 20)
    ### Set parastic capacitances
    CMEMS = 5.4
    Cg = 0.073
    Cpkg = 0.1
    C1 = 1.76
    C2 = C1 / 40
    Cp1 = C1 * 0.12 + Cpkg
    Cp2 = C1 * 0.00 + C2 * 0.00 + Cg
    ### 1st stage signal and noise gain
    Av1 = CMEMS/ (CMEMS + C1 + Cp1)
    Av2 = C1/ C2
    Av_s = Av1* Av2
    Av_n = (C1 * (CMEMS + Cp1)/ (C1 + CMEMS + Cp1) + C2 + Cp2) / C2
    ###1st stage bandwidth
    Bw_1st = 4000
    C3 = 500e-15
    R1 = 1 / (2 * math.pi * Bw_1st * C3)
    ###2nd stage gain
    Av_2nd = 25
    VDD_AMP = 1.2
    Vswing = VDD_AMP / 2 - 0.1
    ADC_max = VDD_AMP / 2
    ADC_bits = 8
    ADC_quant = ADC_max * 2 ** (-ADC_bits + 1)
    audio = audio - np.mean(audio)
    audio_rms = np.sqrt(np.mean(audio**2))
    input_v = audio * (target_sig_vrms / audio_rms)
    noise_v = target_noise_vrms * np.random.randn(np.size(input_v))
    lna_v_pre = Av_s * input_v + Av_n * noise_v
    lna_v = lna_v_pre
    vga_v_pre = Av_2nd * lna_v
    Av_2nd_adj = Av_2nd
    Av_2nd_adj_pre = Av_2nd
    while (np.sum(np.abs(lna_v_pre))> Vswing):
        Av_2nd_adj = Av_2nd_adj - 1
        vga_v_pre = vga_v_pre * Av_2nd_adj / Av_2nd_adj_pre
        Av_2nd_adj_pre = Av_2nd_adj
        if (Av_2nd_adj == 1):
            break
    vga_v = vga_v_pre

    adc_samp = vga_v
    adc_code0 = np.floor(adc_samp / ADC_quant)
    adc_code = adc_code0
    adc_code[adc_code > 2 ** (ADC_bits - 1) - 1] = 2 ** (ADC_bits - 1) - 1
    adc_code[adc_code < -2 ** (ADC_bits - 1)] = -(2 ** (ADC_bits - 1))
    adc_out = adc_code / 2 ** (ADC_bits - 1)
    return  adc_out


