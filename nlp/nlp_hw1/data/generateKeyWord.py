import random

def generateReverseWord(wordlist, number):
    return wordlist + list(reversed(wordlist))[0:number]
def randomWord(wordlist, number):
    return wordlist + random.sample(wordlist, number)
def permuteWord(wordlist):
    random.shuffle(wordlist)
    return wordlist

def modifyData(file, choice, number):
    # input the file name
    f = open(file)
    # output the result
    newfile = open("new_"+str(choice)+"_"+str(number)+"_"+file, 'w')
    for line in f.readlines():
        cur1 = line.strip().split("<EOT>")
        newfile.write(cur1[0] + "<EOT> ")
        # print(cur1[0] + "<EOT>")
        cur2 = (cur1[1]).strip().split("<EOL>")
        wordlist = cur2[0].strip().split(" ")
        # get the candidate wordlist
        if choice == 1:
            candidate = generateReverseWord(wordlist, number)
        elif choice == 2:
            candidate = randomWord(wordlist,number) 
        else : 
            candidate = permuteWord(wordlist)
        newfile.write(" ".join(candidate) + " <EOL>" + cur2[1] + "\n")
    f.close()
    newfile.close()

if __name__ == "__main__":
    md = modifyData
    for i in range(1, 4):
        for j in range(1, 6):
            md("roc.train", i, j)
            md("roc_key.train", i, j)
            md("roc.test", i, j)
            md("roc_key.test", i, j)
