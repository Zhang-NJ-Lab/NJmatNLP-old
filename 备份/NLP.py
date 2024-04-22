import sys
import os
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
        self.actionSave_path.triggered.connect(self.choose_save_path)
        self.actionmodel_umap.triggered.connect(self.model_umap)
        self.actionmodel_plot.triggered.connect(self.model_plot)
        self.actioncosine_similarity.triggered.connect(self.model_cosine_similarity_csv)

        self.tokenizer = BertTokenizerFast.from_pretrained('F:\matbert-cased', do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained('F:\matbert-cased')
        self.reducer = UMAP()
        self.embeddings_umap = None  # 初始化为 None

    def choose_save_path(self):
        save_path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if save_path:
            self.save_path = save_path

    def model_umap(self):
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

        word_to_highlight = "perovskite"
        word5_token = self.tokenizer.encode(word_to_highlight, add_special_tokens=False)
        word5_embedding = self.model.bert.embeddings.word_embeddings(torch.tensor(word5_token)).squeeze().detach().numpy()
        cos_similarities1 = cosine_similarity(self.all_word_embeddings, [word5_embedding])

        plt.figure(figsize=(10, 8))
        cmap = plt.cm.BuPu
        scatter = plt.scatter(self.embeddings_umap[:, 0], self.embeddings_umap[:, 1], c=cos_similarities1.squeeze(), cmap=cmap, s=20)

        highlighted_words = ["perovskite", "lamno3"]
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

        if hasattr(self, 'save_path'):
            plt.savefig(os.path.join(self.save_path, "figure3_word_embeddings_visualization.png"))

        plt.show(block=True)  # 显示图形并阻止程序继续执行


    def model_plot(self):

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
        highlighted_words_blue = ["perovskite", 'CH3NH3PbI3', 'BaTiO3', 'SrTiO3', 'BiFeO3', 'LiNbO3']  # 典型钙钛矿
        highlighted_words_green = ['PEDOT', 'P3HT', 'PCBM']  # 典型HTL材料
        highlighted_words_yellow = ['LiCoO2', 'LiMn2O4', 'LiFePO4']  # 锂电池正极材料
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
        plt.savefig("word_embeddings_visualization.png")
        plt.show()

        if hasattr(self, 'save_path'):
            plt.savefig(os.path.join(self.save_path, "figure3_word_embeddings_highlight.png"))

        plt.show(block=True)  # 显示图形并阻止程序继续执行







    def choose_save_path(self):
        save_path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if save_path:
            self.save_path = save_path

    def model_cosine_similarity_csv(self):
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
        target_word1 = "P1"
        target_token_id = self.tokenizer.encode(target_word1, add_special_tokens=False)
        target_embedding = self.model.bert.embeddings.word_embeddings(
            torch.tensor(target_token_id)).squeeze().detach().numpy()
        cos_similarities2 = cosine_similarity(all_word_embeddings, [target_embedding])
        # 对相似度进行排序并获取排序后的索引
        sorted_indices = np.argsort(cos_similarities2.flatten())[::-1]

        from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
        # 创建或打开 CSV 文件
        if hasattr(self, 'save_path'):
            file_path = os.path.join(self.save_path, 'cosine_similarity_results.csv')
            try:
                # with open(file_path, mode='w', newline='') as file:
                with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                    from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
                    import csv
                    writer = csv.writer(file)
                    writer.writerow(['Rank', 'Word', 'Cosine Similarity'])
                    # 将排名和相似度写入 CSV 文件
                    for rank, idx in enumerate(sorted_indices):
                        word = all_words[idx]
                        similarity = cos_similarities2[idx][0]
                        writer.writerow([rank + 1, word, similarity])
                QMessageBox.information(self, "保存成功", f"CSV 文件已保存到 {file_path}", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存 CSV 文件时出现错误：{str(e)}", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "保存失败", "请先选择保存路径", QMessageBox.Ok)

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






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NLPUI()
    window.show()
    sys.exit(app.exec_())
