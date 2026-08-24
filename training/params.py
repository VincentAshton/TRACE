# Optional imports: some CL methods pull heavy/optional deps (quadprog, qpth, ...).
# Replay only needs base/lora, so make the rest lazy so a missing optional dep
# doesn't break the whole pipeline.
def _try_import(module, name):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception as e:
        print(f"[WARN] optional CL method import failed ({module}.{name}): {e}")
        return None

PP = _try_import("model.Dynamic_network.PP", "PP")
L2P = _try_import("model.Dynamic_network.L2P", "L2P")
LwF = _try_import("model.Regular.LwF", "LwF")
EWC = _try_import("model.Regular.EWC", "EWC")
GEM = _try_import("model.Regular.GEM", "GEM")
OGD = _try_import("model.Regular.OGD", "OGD")
MbPAplusplus = _try_import("model.Replay.MbPAplusplus", "MbPAplusplus")
LFPT5 = _try_import("model.Replay.LFPT5", "LFPT5")
O_LoRA = _try_import("model.Regular.O_LoRA", "O_LoRA")
from model.base_model import CL_Base_Model
from model.lora import lora



Method2Class = {"PP":PP,
                "EWC":EWC,
                "GEM":GEM,
                "OGD":OGD,
                "LwF":LwF,
                "L2P":L2P,
                "MbPA++":MbPAplusplus,
                "LFPT5":LFPT5, 
                "O-LoRA":O_LoRA,
                "base":CL_Base_Model,
                "lora":lora}

AllDatasetName = ["C-STANCE","FOMC","MeetingBank","Papyrus-f","Py150","ScienceQA","ToolBench","NumGLUE-cm","NumGLUE-ds","20Minuten"]

