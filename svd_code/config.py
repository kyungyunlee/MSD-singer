import collections

SVDConfig = collections.namedtuple('SVDConfig', ['sr', 'n_fft', 'hop_len', 'n_mels', 'win_len', 'threshold', 'stride', 'mf_window_size', 'timeLength', 'singingPercent', 'backgroundPercent'])

HPARAMS = SVDConfig(
        sr=16000,
        n_fft=1024,
        hop_len=160,
        n_mels=80,
        win_len=75,
        threshold=0.5,
        stride=1,
        mf_window_size=5,
        timeLength=300,
        singingPercent=0.7,
        backgroundPercent=0.4)

