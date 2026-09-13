"""Render the exact fixed two-label classifier's acceptance-budget example."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(9.4,2.9),gridspec_kw={'width_ratios':[1.25,1]})
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor='none'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
box(.01,.33,.32,.33,'Accept all K\n0 false positives\nSurplus: 0.04','#e4eee8')
box(.55,.58,.42,.32,'Row optimum: accept 1/2 of U\nAdd 0.20 example mass\nSpend: 0.04','#dceaf5')
box(.55,.06,.42,.32,'Exposure optimum: 2/3 of V\nAdd 0.267 positive mass\nSpend: 0.04','#f8e5d3')
for y in [.74,.22]:ax.annotate('',xy=(.54,y),xytext=(.34,.5),arrowprops={'arrowstyle':'->','color':'#4b5563','lw':1.4})
ax.text(.45,.5,'or',ha='center',va='center',fontsize=10)
ax.set_title('One surplus, two competing uses',loc='left',fontsize=11)
bx.bar([-.17,.83],[.6,.5],width=.32,label='Row optimum',color='#4c88b4')
bx.bar([.17,1.17],[8/15,5/9],width=.32,label='Exposure optimum',color='#d88743')
for x,y,txt in [(-.17,.6,'3/5'),(.17,8/15,'8/15'),(.83,.5,'1/2'),(1.17,5/9,'5/9')]:bx.text(x,y+.018,txt,ha='center',fontsize=9)
bx.set_xticks([0,1],['Example coverage','Positive coverage']);bx.set_ylim(0,.75);bx.set_ylabel('Coverage');bx.spines[['top','right']].set_visible(False);bx.legend(loc='upper right',fontsize=8,frameon=False);bx.set_title('Same 90% micro-precision',loc='left',fontsize=11)
fig.tight_layout(w_pad=2);out=R/'paper/figures/utility_mechanism.pdf';fig.savefig(out,bbox_inches='tight');plt.close(fig)
print(out)
