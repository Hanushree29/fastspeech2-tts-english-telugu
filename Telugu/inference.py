import argparse, glob, json, os, unicodedata, yaml, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from scipy.io import wavfile
from utils.model import get_model, get_vocoder, vocoder_infer
from text import text_to_sequence
from text.telugu_text import to_tokens

for f in glob.glob("/usr/share/fonts/**/NotoSansTelugu*Regular*.ttf", recursive=True):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans Telugu", "DejaVu Sans"]

TESTS = {
 "T1":  "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
 "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
 "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
 "T3":  "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
 "T4":  "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}
def lab(t):
    if t == "wb": return "·"
    return "\u25cc" + t if len(t) == 1 and unicodedata.category(t) in ("Mn", "Mc") else t

def bars(vals, toks, ylabel, title, path, color):
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(toks)), 3.6))
    ax.bar(range(len(toks)), vals, color=color)
    ax.set_xticks(range(len(toks))); ax.set_xticklabels([lab(t) for t in toks], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("-p", default="config/Telugu/preprocess.yaml"); ap.add_argument("-m", default="config/Telugu/model.yaml")
    ap.add_argument("-t", default="config/Telugu/train.yaml"); ap.add_argument("--out", default="../Results/Telugu")
    a = ap.parse_args()
    pre, mod, tr = [yaml.load(open(x), Loader=yaml.FullLoader) for x in (a.p, a.m, a.t)]
    cfg = (pre, mod, tr); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(a, cfg, dev, train=False); vocoder = get_vocoder(mod, dev)
    stats = json.load(open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")))
    p_mean, p_std = stats["pitch"][2:4]; e_mean, e_std = stats["energy"][2:4]
    sr, hop = pre["preprocessing"]["audio"]["sampling_rate"], pre["preprocessing"]["stft"]["hop_length"]
    for d in ["test_generated_audios"] + [f"variance_adaptor_plots/{k}" for k in ("duration", "pitch", "energy")]:
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    for sid, text in TESTS.items():
        toks = to_tokens(text)
        seq = np.array([text_to_sequence("{" + " ".join(toks) + "}", ["basic_cleaners"])])
        texts = torch.from_numpy(seq).long().to(dev); lens = torch.tensor([seq.shape[1]]).to(dev)
        with torch.no_grad():
            out = model(torch.tensor([0]).to(dev), texts, lens, seq.shape[1])
            mel, post, p_pred, e_pred, _, d_round, _, _, _, mel_len = out
            wav = vocoder_infer(post.transpose(1, 2), vocoder, mod, pre, lengths=[int(mel_len[0]) * hop])[0]
        wavfile.write(f"{a.out}/test_generated_audios/{sid}.wav", sr, wav)
        n = len(toks)
        pitch = p_pred[0, :n].cpu().numpy() * p_std + p_mean if pre["preprocessing"]["pitch"]["normalization"] else p_pred[0, :n].cpu().numpy()
        energy = e_pred[0, :n].cpu().numpy() * e_std + e_mean if pre["preprocessing"]["energy"]["normalization"] else e_pred[0, :n].cpu().numpy()
        dur_ms = d_round[0, :n].cpu().numpy() * hop / sr * 1000
        bars(dur_ms, toks, "duration (ms)", f"{sid} predicted duration", f"{a.out}/variance_adaptor_plots/duration/{sid}.png", "tab:blue")
        bars(pitch, toks, "F0 (Hz)", f"{sid} predicted pitch", f"{a.out}/variance_adaptor_plots/pitch/{sid}.png", "tab:red")
        bars(energy, toks, "energy", f"{sid} predicted energy", f"{a.out}/variance_adaptor_plots/energy/{sid}.png", "tab:green")
        np.savetxt(f"{a.out}/variance_adaptor_plots/{sid}_values.csv", np.c_[dur_ms, pitch, energy],
                   delimiter=",", header="duration_ms,f0_hz,energy", comments="")
        print(sid, "ok", f"{len(wav)/sr:.2f}s")
