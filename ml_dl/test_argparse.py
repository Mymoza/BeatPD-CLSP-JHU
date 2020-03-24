import argparse
import json

parser = argparse.ArgumentParser(description="Initial Experiment with Wavenet")
parser.add_argument("--data_type",default="cis")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--subtask",default="on_off",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("-uad","--use_ancillarydata",action="store_true")
parser.add_argument("--saveAEFeats",action="store_true")
parser.add_argument("-dlP","--dataLoadParams",type=json.loads)
parser.add_argument("--dataAugScale",default=5,type=int)

args = parser.parse_args()

print(args)

data_type = args.data_type
data_real_subtype = args.data_real_subtype
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata
saveAEFeats = args.saveAEFeats
params = args.dataLoadParams
dataAugScale = args.dataAugScale

#import pdb; pdb.set_trace()
