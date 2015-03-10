
trainfl = '/Volumes/Andy\'s Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Project/Machine-Learning-Natural-Language-Processing/cs4740/old/a3/lib/data/oct27.traindev'
testfl =  '/Volumes/Andy\'s Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Project/Machine-Learning-Natural-Language-Processing/cs4740/old/a3/lib/data/oct27.test'
basefl = '/Volumes/Andy\'s Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Project/Machine-Learning-Natural-Language-Processing/cs4740/old/a3/lib/data/oct27.baseline'
develfl = '/Volumes/Andy\'s Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Project/Machine-Learning-Natural-Language-Processing/cs4740/old/a3/lib/data/oct27.development'
        
def parse_opened_training_file(f):
    # Parses _opened_ file <f> and
    # returns a list of [ list of (POS, word) tuples ]
    data = []
    sentence = []
    for line in f:
        s = line.split()
        if len(s) != 2:
            #print "ERROR:", line
            #exit(-1)
            continue
        if s[0] == "<s>" and len(sentence) > 0:
            data.append(sentence)
            sentence = []
        sentence.append( tuple(s) )
    if len(sentence) > 0:
        data.append(sentence)
    return data
    
def parse_training_file(filename=trainfl):
    # Opens the file and passes it along
    with open(filename,'r') as f:
        return parse_opened_training_file(f)

def parse_development_file(filename=develfl):
    return parse_training_file(filename=filename)

def parse_validation_file(filename=testfl):
    return parse_training_file(filename=filename)

def extract_word_data(data):
    return [ [w for p,w in seq] for seq in data ]

def extract_pos_data(data):
    return [ [p for p,w in seq] for seq in data ]

def parse_opened_test_file(f):
    # Parses _opened_ file <f> and
    # returns a list of [ list of word ]
    data = []
    sentence = []
    for s in f:
        s = s.strip()
        if s == "<s>" and len(sentence) > 0:
            data.append(sentence)
            sentence = []
        sentence.append( s )
    if len(sentence) > 0:
        data.append(sentence)
    return data

def parse_test_file(filename=testfl):
    # Opens the file and passes it along
    with open(filename,'r') as f:
        return parse_opened_test_file(f)