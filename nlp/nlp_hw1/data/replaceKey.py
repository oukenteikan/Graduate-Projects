import pickle

with open('./orderedSentenceEmbeddings_noOverlap', 'rb') as f:
    obj = pickle.load(f)

print(type(obj))
print(len(obj))

def replaceData(file):
    fold = open(file, 'r')
    fnew = open("new_key_no_overlap_" + file, 'w')

    lines = fold.readlines()
    for i in range(len(lines)):
        line = lines[i]
        new_keys = obj[i]
        cur1 = line.strip().split("<EOT>")
        fnew.write(cur1[0] + "<EOT> ")
        cur2 = (cur1[1]).strip().split("<EOL>")
        fnew.write(" ".join(new_keys) + " <EOL>" + cur2[1] + "\n")
    
    fold.close()
    fnew.close()

if __name__ == "__main__":
    rd = replaceData
    rd("roc.train")
    rd("roc_key.train")
    rd("roc.test")
    rd("roc_key.test")

