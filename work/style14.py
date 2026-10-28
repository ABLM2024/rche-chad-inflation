import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'font.size':14,'font.weight':'bold',
 'axes.labelweight':'bold','axes.titleweight':'bold','axes.titlesize':14,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
 'figure.titleweight':'bold','mathtext.fontset':'custom','mathtext.rm':'Liberation Serif:bold','mathtext.it':'Liberation Serif:italic:bold','mathtext.bf':'Liberation Serif:bold',
 'mathtext.default':'bf','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e3e3e0','grid.linewidth':0.8,
 'axes.edgecolor':'#555550','axes.linewidth':1.2,'xtick.major.width':1.2,'ytick.major.width':1.2,'lines.linewidth':2.2,'legend.frameon':False,
 'savefig.dpi':600,'savefig.bbox':'tight','savefig.pad_inches':0.05})
C=dict(prop='#2a78d6',ml='#eb6834',phys='#1baf7a',pers='#eda100',v1='#e87ba4',clim='#9a9a94',obs='#2b2b28',concat='#6250d6')
W=5.8   # text width (in); figures drawn at this width keep 14 pt text at 14 pt in the manuscript
