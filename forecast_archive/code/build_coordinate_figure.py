"""Method schematic with actual saved Foods multipliers (no illustrative outcomes)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
root=Path(__file__).resolve().parents[1]
p=json.loads((root/'evidence/risk_calibration/primary_policy/POLICY_FREEZE.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig=plt.figure(figsize=(10,3.15));a=fig.add_axes([.015,.12,.59,.8]);a.set_xlim(0,10);a.set_ylim(0,4);a.axis('off')
def box(x,y,w,h,text,color='#eef3f7'):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',facecolor=color,edgecolor='#8295a5',linewidth=.8));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5)
def arrow(x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1,color='#405466'))
box(.1,2.5,2.5,1.1,'Past sales and\navailability signals')
box(3.2,2.6,2.1,.9,'Stockout risk q(X)')
box(6,2.4,3.7,1.3,'Product group × risk bin\nLook up learned multiplier', '#e4f0ee')
box(.1,.2,2.5,1.1,'Any nonnegative\nbase forecast b(X)')
box(6,.25,3.7,1.1,'Multiply: f(X) = s × b(X)\nEvery row keeps a forecast','#e4f0ee')
arrow(2.7,3.05,3.1,3.05);arrow(5.4,3.05,5.9,3.05);arrow(7.85,2.3,7.85,1.4);arrow(2.7,.75,5.9,.75)
a.text(4.2,1.7,'Fit s on separate calibration targets;\nshrink toward the group-only scale',ha='center',va='center',fontsize=8,color='#405466')
a.set_title('(a) Risk supplies a coordinate, not an uplift rule',loc='left',fontsize=10,pad=10)
b=fig.add_axes([.69,.24,.29,.59]); cs=p['base_category_scales']['FOODS'];rat=[p['risk_policy']['scales'][f'FOODS|{j}']/cs for j in range(8)]
b.axhline(1,color='#66717d',ls='--',lw=1);b.plot(range(1,9),rat,'o-',color='#147e79',lw=1.7,ms=4)
b.set(xticks=range(1,9),xlabel='Risk bin (low to high)',ylabel='Multiplier / group scale',ylim=(-.05,1.22));b.spines[['top','right']].set_visible(False)
b.text(1.3,.18,'Suppression',color='#405466',fontsize=8);b.text(6.0,1.14,'Uplift',color='#405466',fontsize=8)
b.set_title('(b) Actual M5 Foods policy',fontsize=10,pad=12)
(root/'paper/figures').mkdir(exist_ok=True)
fig.savefig(root/'paper/figures/coordinate_method.pdf',bbox_inches='tight');fig.savefig(root/'paper/figures/coordinate_method.png',dpi=180,bbox_inches='tight');plt.close(fig)
