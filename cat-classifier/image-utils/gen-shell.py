
import sys

def main(argv):
    count = 1
    f = open(argv[1], "r")
    for line in f:
        print("echo \' executing task  "+str(count)+" \' ")
        print(argv[2] +" " +line.rstrip() + " " + argv[3])
        print("echo \' \' ")
        count += 1
        print('')

if __name__== "__main__":
    sys.argv[0]
    main(sys.argv)


"""
e.g usage

python /Users/camel/d2/Work/codebarn/mackerel/image-utils/gen-shell.py /tmp/batch.list  \
 ' python /Users/camel/d2/Work/codebarn/mackerel/image-utils/resize-img.py '   \
 ' 64 64 /Volumes/D2/Tuna/cat-or-dog/train-64x64' > /tmp/dow 
"""

