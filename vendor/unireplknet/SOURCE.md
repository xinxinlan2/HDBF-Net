# UniRepLKNet source

Upstream: https://github.com/AILab-CVC/UniRepLKNet
File: https://raw.githubusercontent.com/AILab-CVC/UniRepLKNet/main/unireplknet.py
Retrieved: 2026-11-05. License: Apache-2.0, included.

Local numerical change: GRN reductions and affine computations use FP32 and cast
back to the input dtype; state dictionary names/shapes are unchanged.
Optional iGEMM and framework checkpoint auto-loading are disabled by the wrapper.

Checkpoint: upernet_unireplknet_s_in22k_512_160k_ade20k_miou51.86.pth
Original: https://huggingface.co/DingXiaoH/UniRepLKNet
Size: 1059007654 bytes
SHA256: 67efb2494f12e2bccaa19282b8d2eb0d466fe5755584e5fb9903cd188ebc0e64
Mirror download verified against the published LFS pointer.
