"""Render the actual model information flow as editable vector artwork."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent


def main():
    fig,ax=plt.subplots(figsize=(13,6.2))
    ax.set(xlim=(0,13),ylim=(0,6.2));ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.10",
                                   facecolor=color,edgecolor="#3f4246",linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    def arrow(a,b,style="-"):
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#242629",
                                               "linewidth":1.2,"linestyle":style})
    box(.2,2.5,1.6,1.0,"SAR PRE1/2/3\nSAR POST1\nVV / VH", "#cedde6")
    box(2.3,2.4,1.6,1.2,"Trainable\nSAR U-Net\nencoder", "#a9c6d7")
    box(4.5,2.4,2.,1.2,"SAR memory\nstrides 4 / 8 / 16\n128 channels", "#aac8b4")
    box(4.5,4.8,2.,.9,"Optical PRE/POST\nshared change encoder", "#ead39c")
    box(7.2,4.8,1.8,.9,"Teacher policy\noffsets / weights", "#e4c078")
    box(7.2,.6,1.8,.9,"Student policy\nSAR input only", "#a9c6d7")
    box(7.4,2.5,1.5,1.0,"Deformable\nSAR retrieval", "#aac8b4")
    box(9.6,2.5,1.4,1.0,"Trainable\nSAR decoder", "#a9c6d7")
    box(11.6,2.5,1.1,1.0,"Landslide\nmask", "#d6d9de")
    arrow((1.8,3),(2.3,3));arrow((3.9,3),(4.5,3))
    arrow((6.5,3),(7.4,3));arrow((8.9,3),(9.6,3));arrow((11,3),(11.6,3))
    arrow((6.5,5.25),(7.2,5.25));arrow((6.5,3.5),(7.3,4.8))
    arrow((6.5,2.5),(7.2,1.1));arrow((8.1,4.8),(8.1,3.5))
    arrow((8.1,1.5),(8.1,2.5))
    ax.plot([3.1,3.1,10.2,10.2],[2.4,1.95,1.95,2.5],color="#242629",linewidth=1.1)
    ax.text(4.5,2.05,"SAR high-resolution skips",fontsize=9)
    arrow((8.9,4.8),(8.9,1.5),"--")
    ax.text(9.05,4.15,"Policy JS\n(shared or scale-aware)",fontsize=9)
    ax.text(9.05,.8,"Evidence cosine\non the same detached\nStudent SAR memory",fontsize=9)
    ax.text(.3,5.8,"Optical-guided Teacher and SAR-only Student use separate forward paths",
            fontsize=13,fontweight="bold")
    ax.text(.3,.18,"M1: explicit SAR-change memory   |   M2: scale-aware JS   |   M3: Haar query context   |   M4: SAR channel gains",
            fontsize=10)
    fig.tight_layout()
    for extension in ("svg","pdf","png"):
        fig.savefig(HERE/f"architecture.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
