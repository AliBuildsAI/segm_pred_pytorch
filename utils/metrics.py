import torchnet as tnt
import math

class sem_segm_meter(tnt.meter.ConfusionMeter):
	def __init__ (self,classes):
		super(sem_segm_meter,self).__init__(len(classes))
		self.classes=classes
		self.nclasses=len(classes)
		#self.conf=tnt.meter.ConfusionMeter(self.nclasses)
		self.totalValid = 0
		self.averageValid = 0
		self.valids = [0]*self.nclasses
		self.unionvalids = [0]*self.nclasses
        
	def avgIOU(self):
		total=0
		for t in range(self.nclasses):
			self.valids[t] = self.conf[t,t] / self.conf.sum(0)[t]
			self.unionvalids[t] = self.conf[t,t] / (self.conf.sum(0)[t]+self.conf.sum(1)[t]-self.conf[t,t])
			total = total + self.conf[t,t]
		    
		self.totalValid=sum(self.conf.diagonal())/self.conf.sum()
		self.averageValid = 0
		self.averageUnionValid = 0
		nvalids = 0
		nunionvalids = 0
		for t in range(self.nclasses):
			if (not math.isnan(self.valids[t])):
				self.averageValid = self.averageValid + self.valids[t]
				nvalids = nvalids + 1
			if (not math.isnan(self.valids[t]) and not math.isnan(self.unionvalids[t])):
				self.averageUnionValid = self.averageUnionValid + self.unionvalids[t]
				nunionvalids = nunionvalids + 1
		self.averageValid = self.averageValid / nvalids
		self.averageUnionValid = self.averageUnionValid / nunionvalids
		return self.averageUnionValid*100
