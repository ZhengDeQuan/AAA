import argparser

def main(args):
    train_file =

if __name__ == "__main__":
    parser = argparser.ArgumentParser(
        description=__doc__,
        formatter_class = argparser.RawDescriptionHelpFormatter
    )
    parser.add_argument('-train_file','--train_file',dtype=str,default="/home4/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_continues.txt")
    parser.add_argument('-test_file','--test_file',dtype=str,default="/home4/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_continues.txt")
    args = parser.parse_args()