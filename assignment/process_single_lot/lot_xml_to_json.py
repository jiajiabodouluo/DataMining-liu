# This is taken directly from resources, with only the file names changed

from xmljson import parker, Parker
from xmljson import yahoo
from xml.etree.ElementTree import fromstring
from json import dumps,loads
import pprint as pp


def xmltojson(file_name,out_file=None):
    fp = open(file_name,'r')

    xmldata = fp.read()

    jsond = dumps(yahoo.data(fromstring(xmldata)))

    jsond = loads(jsond)

    spaces = jsond['parking']['space']

    if not out_file is None:
        f = open(out_file,'w')
        f.write(dumps(spaces,indent=4, separators=(',', ': ')))
        f.close()

    for space in spaces:
        print(space['contour'])
        for point in space['contour']['point']:
            print(point)

# Literally the only change is the files' names
if __name__=='__main__':
    filename = '2012-12-16_12_05_07.xml'
    xmltojson(filename,'2012-12-16_12_05_07.json',)
