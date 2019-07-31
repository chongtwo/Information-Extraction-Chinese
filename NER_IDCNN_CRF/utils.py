import os
import json
import shutil
import logging
import datetime
import codecs
import tensorflow as tf
from conlleval import return_report

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = logging.getLogger(log_file+time)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# def test_ner(results, path):
#     """
#     Run perl script to evaluate model
#     """
#     script_file = "conlleval"
#     output_file = os.path.join(path, "ner_predict.utf8")
#     result_file = os.path.join(path, "ner_result.utf8")
#     with open(output_file, "w") as f:
#         to_write = []
#         for block in results:
#             for line in block:
#                 to_write.append(line + "\n")
#             to_write.append("\n")
#
#         f.writelines(to_write)
#     os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
#     eval_lines = []
#     with open(result_file) as f:
#         for line in f:
#             eval_lines.append(line.strip())
#     return eval_lines


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def create_model(session, Model_class, path, load_vec, config, id_to_char, logger, is_train=True):
    # create model, reuse parameters if exists
    model = Model_class(config, is_train)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value())
            emb_weights = load_vec(config["emb_file"],id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


def result_to_json(string, tags, tag_schema):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    if(tag_schema== "iob"):
        last_tag = "O"
        for char, tag in zip(string, tags):
            if tag[0] == "B":
                if last_tag[0] != "O":  # BI BI.../B BI...
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                entity_name = char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            else: #tag[0] == "哦"
                if last_tag != "O":
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                    entity_start = idx
            idx += 1
            last_tag = tag
    elif(tag_schema == "iobes"):
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
    else:
        ValueError("未知的tag_schema")
    return item

if __name__ == "__main__":
    f_path = "F:\肺癌分期项目程序\标注数据\CT_SCHEMA2_BIO_2\CT_FIND_0003.bio"
    filname = f_path.split("\\")[-1].replace(".bio",".json")
    o_path = "F:/肺癌分期项目程序/ner_result/"+filname
    with codecs.open(f_path, "r", encoding="utf8") as f:
        with codecs.open(o_path, "w", encoding="utf8") as outf:
            lines = f.readlines()
            string = ""
            tags=[]
            result_arr = []
            i = 0
            for line in lines:
                print(i)
                i = i+1
                if line != "\r\n" and line !="\n":
                    string += line.split(" ")[0]
                    tags.append(line.split(" ")[1].strip("\n"))
                else:
                    result = result_to_json(string, tags, "iob")
                    result_arr.append(result)
                    string=""
                    tags=[]
            json.dump(result_arr, outf, ensure_ascii=False)




