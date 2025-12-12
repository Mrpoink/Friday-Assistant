

#So we need to do these things with these requirements:
#Build a class with the sizes of the datasets (rows)
#Take in the total amount of rows
#Our goal is to figure out the most optimal way to split up the datasets using these requirements
#We must have 6 epochs
#We have roughly 12 datasets
#We need to Put them into their relative pools for each epoch
#All values in the rows must be used
#The number of rows for each epoch is subjective
#We must keep the flow of Friday standard (How we want her to be)
#Epoch 1-2 are for Identity
#Epoch 3-4 are for Tools
#Epoch 5-6 are for Intelligence

#Datasets for the Identity/Creative pool:
#Identity_hh 1, 2, 3, 4, 5, 6
#SMS 1, 2, 3, 4, 5, 6
#Airboros 2
#Dolphin 2
#
#Datasets for Tools pool:
#Tools (glaive) 4
#Bespoke (RAG) 3, 4
#Empath?? 3
#OpenThoughts 4, 5, 6
#
#Datasets for Intelligence pool:
#Enigmata 5, 6
#OpenMix 5, 6
#Magicode 6


class DatasetSplitter:
    
    def __init__(self):
        self.total_size = 0 #Enter the real value of total rows
        self.epochs = 6
        self.dataset_appears = {
            # === ANCHORS (The Soul - Always Present) ===
            'Identity_hh' : {
                'appears' : [1, 2, 3, 4, 5, 6],
                'size' : 17280,
                'all' : True,
                'Double' : False
            },
            'SMS' : {
                'appears': [1, 2, 3, 4, 5, 6],
                'size' : 27000,
                'all' : True,
                'Double' : False
            },

            # === INTELLIGENCE (The Brain - Epochs 1-2) ===
            # Moved from 6 -> 1 to establish coding syntax first
            'Magicode' : {
                'appears' : [1],
                'size' : 34560,
                'all' : False,
                'Double' : True
            },
            # Moved from 5,6 -> 1,2 to build general smarts
            'OpenMix' : {
                'appears' : [1, 2],
                'size' : 46080,
                'all' : False,
                'Double' : True
            },
             # Moved from 4,5,6 -> 1,2,3 to act as the "reasoning spine"
            'OpenThoughts' : {
                'appears' : [1, 2, 3],
                'size' : 29543,
                'all' : False,
                'Double' : True
            },

            # === TOOLS & TRANSITION (The Hands - Epochs 3-4) ===
            # Moved from 5,6 -> 2,3 to bridge Logic into Action
            'Enigmata' : {
                'appears' : [2, 3], 
                'size' : 12000,
                'all' : True,
                'Double' : False
            },
            # Moved from 4 -> 3 to centralize tool learning
            'Tools': {
                'appears' : [3],
                'size' : 28800,
                'all' : False,
                'Double' : False
            },
            # Kept in 3,4 as it bridges Tools and Creative
            'Bespoke' : {
                'appears':[3, 4],
                'size' : 13341,
                'all' : False,
                'Double' : False
            },

            # === CREATIVE & IDENTITY (The Personality - Epochs 5-6) ===
            # Moved from 2 -> 4,5 to introduce creative writing style
            'Airboros': {
                'appears':[4, 5],
                'size' : 14400,
                'all' : False,
                'Double' : False
            },
            # Moved from 2 -> 5 for "Chat" style
            'Dolphin' : {
                'appears' : [5],
                'size' : 14400,
                'all' : False,
                'Double' : False
            },
            # Moved from 3 -> 6 for the final emotional polish
            'Empath' : {
                'appears' : [6],
                'size' : 12000,
                'all' : False,
                'Double' : False
            }
        }
        self.goal_datasets = {
            'Epoch 1 - 2': [], #Identity/Creative pool
            'Epoch 3 - 4': [], #Tools pool
            'Epoch 5 - 6': [] #Intelligence pool
            } #Same thing as above, but for the goal datasets per epoch
        
        self.epoch_size = self.total_size / self.epochs #Goal size for each epoch
        
    def section_out_by_epoch(self):
        self.standard_epoch_vals = {}
        full_size = 0
        for i in range(1, 7):
            datasets_in_epoch = {}
            total_for_epoch = 0
            for dataset, info in self.dataset_appears.items():
                if i in info['appears']:
                    datasets_in_epoch[dataset] = info['size'] / len(info['appears'])
                    total_for_epoch += info['size'] / len(info['appears'])
                    
            full_size += total_for_epoch
                
            self.standard_epoch_vals[i] = {'datasets':datasets_in_epoch, 'full_size' : total_for_epoch}
            
        return self.standard_epoch_vals, full_size
    
    def percentages(self):
        pass
    
    def different_section_all(self):
        self.standard_epoch_vals = {}
        full_size = 0
        for i in range(6):
            epoch_size = 0
            datasets_in_epoch = {}
            
            for dataset, info in self.dataset_appears.items():
                if info['all'] == False:
                    if info['Double'] == True: 
                        datasets_in_epoch[dataset] = (info['size'] / self.epochs) * 2
                        epoch_size += (info['size'] / self.epochs) * 2
                    if info['Double'] == False:
                        datasets_in_epoch[dataset] = info['size'] / self.epochs
                        epoch_size += info['size'] / self.epochs
                if info['all'] == True:
                    datasets_in_epoch[dataset] = info['size']
                    epoch_size += info['size']
            
            full_size += epoch_size
            
            self.standard_epoch_vals[i] = {'datasets':datasets_in_epoch, 'full_size' : epoch_size}
            
        return self.standard_epoch_vals, full_size
    
    
    
splitter = DatasetSplitter()

standard, size = splitter.different_section_all()

for epoch, info in standard.items():
    print(f"FOR EPOCH: {epoch}\nINFO:\n{info['datasets']}\n{info['full_size']}\n=============")

print(f"SIZE FOR ENTIRE TRAINING RUN: {size}")        
        
       
            
                
            