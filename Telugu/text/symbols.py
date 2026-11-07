
_pad = "_"
_special = ["sil", "sp", "spn", "wb"]
_telugu = [chr(c) for c in range(0x0C00, 0x0C80)]
symbols = [_pad] + ["@" + s for s in _special + _telugu]
