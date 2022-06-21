import sys
import getopt

def input(argv):
    arg_lr = "0.0001"
    arg_gamma = "0.99"
    arg_help = "{0} --lr <learning-rate> --gamma <gamma>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "h:l:g:", ["help", "lr=",
                                                         "gamma="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-l", "--lr"):
            arg_lr = arg
        elif opt in ("-g", "--gamma"):
            arg_gamma = arg

    return arg_lr, arg_gamma