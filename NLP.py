import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import torch
import numpy as np
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # 使用 Qt5 作为后端

class NLPUI(QMainWindow):
    def __init__(self):
        super(NLPUI, self).__init__()
        loadUi("mainwindow.ui", self)
        self.actionmodel_path.triggered.connect(self.choose_model_path)
        self.actionmodel_umap.triggered.connect(self.show_model_umap)
        self.actionmodel_plot.triggered.connect(self.show_model_plot)
        self.actioncosine_similarity.triggered.connect(self.model_cosine_similarity_csv)
        self.actionSave_path.triggered.connect(self.choose_save_path)  # 绑定保存路径按钮
        # csv_clean
        self.actioncsv_clean.triggered.connect(self.csv_clean)
        self.tokenizer = None
        self.model = None
        self.model_path = None
        self.save_path = None
        self.reducer = UMAP()
        self.embeddings_umap = None  # 初始化为 None

        #逻辑
        self.choose_model_path_state=False
        self.choose_save_path_state = False




    def choose_model_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        model_path = QFileDialog.getExistingDirectory(self, "选择模型文件夹路径", options=options)
        if model_path:
            self.model_path = model_path
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path, do_lower_case=False)
            self.model = BertForMaskedLM.from_pretrained(self.model_path)
            self.choose_model_path_state = True
            print(self.choose_model_path_state)

    def choose_save_path(self):
        save_path = QFileDialog.getExistingDirectory(self, "选择保存路径")

        if save_path:
            self.save_path = save_path
            self.choose_save_path_state =True









    # ------------------------------------------------------------------------



















    def model_cosine_similarity_csv(self):

        value, ok = QtWidgets.QInputDialog.getText(self, "cosine similarity ranking", "", QtWidgets.QLineEdit.Normal, "water")
        all_word_embeddings = []
        all_words = []
        for word in self.tokenizer.vocab.keys():
            if word.isalnum():
                token_id = self.tokenizer.encode(word, add_special_tokens=False)
                if len(token_id) == 1:
                    token_id = torch.tensor(token_id).unsqueeze(0)
                    embeddings = self.model.bert.embeddings.word_embeddings(token_id)
                    all_word_embeddings.append(embeddings.squeeze().detach().numpy())
                    all_words.append(word)

        all_word_embeddings = np.array(all_word_embeddings)
        self.all_word_embeddings = all_word_embeddings
        self.all_words = all_words
        self.embeddings_umap = self.reducer.fit_transform(all_word_embeddings)

        # 计算目标词与所有其他词的余弦相似度
                                                   #
        target_word1 = value
        target_token_id = self.tokenizer.encode(target_word1, add_special_tokens=False)
        target_embedding = self.model.bert.embeddings.word_embeddings(
            torch.tensor(target_token_id)).squeeze().detach().numpy()
        cos_similarities2 = cosine_similarity(all_word_embeddings, [target_embedding])
        # 对相似度进行排序并获取排序后的索引
        sorted_indices = np.argsort(cos_similarities2.flatten())[::-1]

        from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
        # 创建或打开 CSV 文件
        import shutil
        mat_path = self.save_path + "/matbert"
        if os.path.exists(mat_path):
            shutil.rmtree(mat_path)
        os.makedirs(mat_path)
        file_path = os.path.join(mat_path, 'cosine_similarity_results.csv')
        try:
            # with open(file_path, mode='w', newline='') as file:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
                import csv
                writer = csv.writer(file)
                # writer.writerow(['Rank', 'Word', 'Cosine Similarity'])
                writer.writerow(['Word', 'Cosine Similarity'])
                # 将排名和相似度写入 CSV 文件
                for rank, idx in enumerate(sorted_indices):
                    word = all_words[idx]
                    similarity = cos_similarities2[idx][0]
                    # writer.writerow([rank + 1, word, similarity])
                    writer.writerow([word, similarity])
            QMessageBox.information(self, "保存成功", f"CSV 文件已保存到 {file_path}", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存 CSV 文件时出现错误：{str(e)}", QMessageBox.Ok)
        str1 = (mat_path + '/cosine_similarity_results.csv').replace("/", "\\")
        os.startfile(str1)


        # 展示图片的代码可以在这里添加

    #
    # def model_cosine_similarity_csv(self):
    #
    #     all_word_embeddings = []
    #     all_words = []
    #     for word in self.tokenizer.vocab.keys():
    #         if word.isalnum():
    #             token_id = self.tokenizer.encode(word, add_special_tokens=False)
    #             if len(token_id) == 1:
    #                 token_id = torch.tensor(token_id).unsqueeze(0)
    #                 embeddings = self.model.bert.embeddings.word_embeddings(token_id)
    #                 all_word_embeddings.append(embeddings.squeeze().detach().numpy())
    #                 all_words.append(word)
    #
    #     all_word_embeddings = np.array(all_word_embeddings)
    #     self.all_word_embeddings = all_word_embeddings
    #     self.all_words = all_words
    #     self.embeddings_umap = self.reducer.fit_transform(all_word_embeddings)  # 更新 embeddings_umap
    #
    #     import csv
    #
    #     # 定义要比较的词
    #     target_word1 = "P1"
    #     # 计算目标词与所有其他词的余弦相似度
    #     target_token_id = self.tokenizer.encode(target_word1, add_special_tokens=False)
    #     target_embedding = self.model.bert.embeddings.word_embeddings(
    #         torch.tensor(target_token_id)).squeeze().detach().numpy()
    #     cos_similarities2 = cosine_similarity(all_word_embeddings, [target_embedding])
    #     # 对相似度进行排序并获取排序后的索引
    #     sorted_indices = np.argsort(cos_similarities2.flatten())[::-1]
    #
    #     # 创建或打开 CSV 文件
    #     with open('cosine_similarity_results.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Rank', 'Word', 'Cosine Similarity'])
    #         # 将排名和相似度写入 CSV 文件
    #         for rank, idx in enumerate(sorted_indices):
    #             word = all_words[idx]
    #             similarity = cos_similarities2[idx][0]
    #             writer.writerow([rank + 1, word, similarity])
    #
    #     if hasattr(self, 'save_path'):
    #         plt.savefig(os.path.join(self.save_path, 'cosine_similarity_results.csv'))
    #
    #     plt.show(block=True)  # 显示图形并阻止程序继续执行

# 开始写传参=-------------------------------------------
    def matbert_model_umap(self, target_word,highlight_words):

        try:
            def model_umap(target_word, highlight_words, mat_path):
                if not self.tokenizer or not self.model:
                    from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
                    QMessageBox.warning(self, "未选择模型文件", "请先选择模型文件", QMessageBox.Ok)
                    return

                all_word_embeddings = []
                all_words = []
                for word in self.tokenizer.vocab.keys():
                    if word.isalnum():
                        token_id = self.tokenizer.encode(word, add_special_tokens=False)
                        if len(token_id) == 1:
                            token_id = torch.tensor(token_id).unsqueeze(0)
                            embeddings = self.model.bert.embeddings.word_embeddings(token_id)
                            all_word_embeddings.append(embeddings.squeeze().detach().numpy())
                            all_words.append(word)

                all_word_embeddings = np.array(all_word_embeddings)
                self.all_word_embeddings = all_word_embeddings
                self.all_words = all_words
                self.embeddings_umap = self.reducer.fit_transform(all_word_embeddings)  # 更新 embeddings_umap

                word_to_highlight = target_word[0]
                word5_token = self.tokenizer.encode(word_to_highlight, add_special_tokens=False)
                word5_embedding = self.model.bert.embeddings.word_embeddings(torch.tensor(word5_token)).squeeze().detach().numpy()
                cos_similarities1 = cosine_similarity(self.all_word_embeddings, [word5_embedding])

                plt.figure(figsize=(10, 8))
                cmap = plt.cm.BuPu
                scatter = plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], c=cos_similarities1.squeeze(), cmap=cmap, s=20)

                highlighted_words = highlight_words
                for word in highlighted_words:
                    if word in self.all_words:
                        idx = self.all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='red', marker='*', s=100)
                        plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='red')

                plt.title('UMAP Visualization of BERT Word Embeddings')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')

                cbar = plt.colorbar(scatter)
                cbar.set_label('Cosine Similarity')
                cbar.set_ticks(np.linspace(0, 1, 6))

                #if hasattr(self, mat_path):
                plt.savefig(os.path.join(mat_path, "figure3_word_embeddings_umap.png"))
                # plt.show(block=True)  # 显示图形并阻止程序继续执行

                plt.figure(figsize=(10, 8))
                cmap = plt.cm.BuPu
                scatter = plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], c=cos_similarities1.squeeze(), cmap=cmap, s=20)

                highlighted_words = highlight_words
                for word in highlighted_words:
                    if word in self.all_words:
                        idx = self.all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='red', marker='*', s=100)
                        # plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='red')

                plt.title('UMAP Visualization of BERT Word Embeddings')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')

                cbar = plt.colorbar(scatter)
                cbar.set_label('Cosine Similarity')
                cbar.set_ticks(np.linspace(0, 1, 6))

                # if hasattr(self, mat_path):
                plt.savefig(os.path.join(mat_path, "figure3_without_word_embeddings_umap.png"))



            import shutil
            mat_path = self.save_path + "/matbert"
            if os.path.exists(mat_path):
                shutil.rmtree(mat_path)
            os.makedirs(mat_path)

            self.di.close()


            model_umap(target_word,highlight_words, mat_path)
            # # 将模型传递给 plot_word_vectors_highlighted 函数
            # plot_word_vectors_highlighted(model, highlight_words_blue, highlight_words_red, highlight_words_yellow,
            #                               highlight_words_green, highlight_words_orange)

            str1 = (mat_path + '/figure3_word_embeddings_umap.png').replace("/", "\\")
            os.startfile(str1)

            # figure3_without_word_embeddings_umap.png
            str2 = (mat_path + '/figure3_without_word_embeddings_umap.png').replace("/", "\\")
            os.startfile(str2)


            from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)

        except Exception as e:
            print(e)

    def show_model_umap(self):
        def give(a, b):
            target_word = a.split(',')
            highlight_words = b.split(',')

            self.matbert_model_umap(target_word,highlight_words)

        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
            if self.choose_model_path_state==True and self.choose_save_path_state ==True:
                import dialog_model_umap
                #from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QtWidgets
                self.di = QtWidgets.QDialog()
                d = dialog_model_umap.Ui_Dialog()
                d.setupUi(self.di)
                self.di.show()
                d.buttonBox.accepted.connect(
                    lambda: give(d.lineEdit.text(), d.lineEdit_2.text()))
                d.buttonBox.rejected.connect(self.di.close)
            elif self.choose_model_path_state==False:
                QMessageBox.information(self, 'Hint', 'Do matbert-model--model path!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            elif self.choose_save_path_state == False:
                QMessageBox.information(self, 'Hint', 'Do Save path!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
        except Exception as e:
            print(e)

    def matbert_model_plot(self, highlighted_words_blue_enter,highlighted_words_green_enter,highlighted_words_yellow_enter):

        try:
            def model_plot(highlighted_words_blue_enter,highlighted_words_green_enter,highlighted_words_yellow_enter,mat_path):
                all_word_embeddings = []
                all_words = []
                for word in self.tokenizer.vocab.keys():
                    if word.isalnum():
                        token_id = self.tokenizer.encode(word, add_special_tokens=False)
                        if len(token_id) == 1:
                            token_id = torch.tensor(token_id).unsqueeze(0)
                            embeddings = self.model.bert.embeddings.word_embeddings(token_id)
                            all_word_embeddings.append(embeddings.squeeze().detach().numpy())
                            all_words.append(word)

                all_word_embeddings = np.array(all_word_embeddings)
                self.all_word_embeddings = all_word_embeddings
                self.all_words = all_words
                self.embeddings_umap = self.reducer.fit_transform(all_word_embeddings)  # 更新 embeddings_umap

                plt.figure(figsize=(10, 8))

                # 先绘制灰色的点
                plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], color='gray', s=20)

                # 高亮的词语及其对应的位置
                highlighted_words_blue = highlighted_words_blue_enter  # 典型钙钛矿
                highlighted_words_green = highlighted_words_green_enter  # 典型HTL材料
                highlighted_words_yellow = highlighted_words_yellow_enter # 锂电池正极材料
                ''
                for word in highlighted_words_blue:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='blue', s=20)
                        plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='blue')

                for word in highlighted_words_green:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='green', s=20)
                        plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='green')

                for word in highlighted_words_yellow:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='yellow', s=20)
                        plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='yellow')

                plt.title('UMAP Visualization of BERT Word Embeddings')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')

                plt.savefig(os.path.join(mat_path, "figure3_word_embeddings_highlight.png"))

                #plt.show(block=True)  # 显示图形并阻止程序继续执行
                plt.figure(figsize=(10, 8))

                # 先绘制灰色的点
                plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], color='gray', s=20)

                # 高亮的词语及其对应的位置
                highlighted_words_blue = highlighted_words_blue_enter  # 典型钙钛矿
                highlighted_words_green = highlighted_words_green_enter  # 典型HTL材料
                highlighted_words_yellow = highlighted_words_yellow_enter # 锂电池正极材料
                ''
                for word in highlighted_words_blue:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='blue', s=20)
                        # plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='blue')

                for word in highlighted_words_green:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='green', s=20)
                        # plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='green')

                for word in highlighted_words_yellow:
                    if word in all_words:
                        idx = all_words.index(word)
                        plt.scatter(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], color='yellow', s=20)
                        # plt.text(self.embeddings_umap[idx, 0], self.embeddings_umap[idx, 1], word, fontsize=8, ha='left', va='bottom', color='yellow')

                plt.title('UMAP Visualization of BERT Word Embeddings')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')

                plt.savefig(os.path.join(mat_path, "figure3_word_embeddings_highlight_without_word.png"))



            import shutil
            mat_path = self.save_path + "/matbert"
            if os.path.exists(mat_path):
                shutil.rmtree(mat_path)
            os.makedirs(mat_path)

            self.di.close()


            model_plot(highlighted_words_blue_enter,highlighted_words_green_enter,highlighted_words_yellow_enter, mat_path)
            # # 将模型传递给 plot_word_vectors_highlighted 函数
            # plot_word_vectors_highlighted(model, highlight_words_blue, highlight_words_red, highlight_words_yellow,
            #                               highlight_words_green, highlight_words_orange)

            str1 = (mat_path + '/figure3_word_embeddings_highlight.png').replace("/", "\\")
            os.startfile(str1)
            str2 = (mat_path + '/figure3_word_embeddings_highlight_without_word.png').replace("/", "\\")
            os.startfile(str2)



            from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)

        except Exception as e:
            print(e)

    def show_model_plot(self):
        def give(a, b,c):
            highlighted_words_blue_enter=a.split(',')
            highlighted_words_green_enter=b.split(',')
            highlighted_words_yellow_enter=c.split(',')


            self.matbert_model_plot(highlighted_words_blue_enter,highlighted_words_green_enter,highlighted_words_yellow_enter)

        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
            if self.choose_model_path_state==True and self.choose_save_path_state ==True:
                import dialog_model_plot
                #from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QtWidgets
                self.di = QtWidgets.QDialog()
                d = dialog_model_plot.Ui_Dialog()
                d.setupUi(self.di)
                self.di.show()
                d.buttonBox.accepted.connect(
                    lambda: give(d.lineEdit.text(), d.lineEdit_2.text(),d.lineEdit_3.text()))
                d.buttonBox.rejected.connect(self.di.close)
            elif self.choose_model_path_state==False:
                QMessageBox.information(self, 'Hint', 'Do matbert-model--model path!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            elif self.choose_save_path_state == False:
                QMessageBox.information(self, 'Hint', 'Do Save path!', QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
        except Exception as e:
            print(e)









    def csv_clean(self):

        import shutil
        mat_path = self.save_path + "/matbert"
        if os.path.exists(mat_path):
            shutil.rmtree(mat_path)
        os.makedirs(mat_path)


        try:
            file_path, _ = QFileDialog.getOpenFileName(self, '选择 CSV 文件', '', 'CSV files (*.csv)')

            import pandas as pd
            from chemdataextractor import Document

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 提取第一列数据
            column_data = df.iloc[:, 0]

            # 定义函数来识别化学物质
            def is_chemical(entity):
                doc = Document(entity)
                chem_entities = doc.cems
                return len(chem_entities) > 0

            # 创建新列来表示化学物质
            df['Chemical'] = column_data.apply(lambda x: 1 if is_chemical(x) else 0)

            # 删除最后一列是 0 的整行
            df = df[df.iloc[:, -1] != 0]

            # 删除最后一列
            df = df.drop(df.columns[-1], axis=1)

            # 保存结果到新的CSV文件，命名为 clean.csv
            df.to_csv(mat_path + "/clean.csv", index=False)

            str1 = (mat_path + '/clean.csv').replace("/", "\\")
            os.startfile(str1)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NLPUI()
    window.show()
    sys.exit(app.exec_())
