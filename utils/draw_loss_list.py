import matplotlib.pyplot as plt


def draw_loss_list(train_loss, val_loss, save=True, epoch=-1):
    """epoch: num of complete epochs"""

    if epoch > -1:
        train_loss = train_loss[: epoch + 1]
        val_loss = val_loss[: epoch + 1]

    def draw_one_list(loss_list, title):
        plt.plot(loss_list)
        plt.title(title)
        plt.ylabel("loss")
        plt.xlabel("train epochs")

    plt.figure()
    plt.subplot(2, 1, 1)
    draw_one_list(train_loss, "train loss list")
    plt.subplot(2, 1, 2)
    draw_one_list(val_loss, "val loss list")
    plt.tight_layout()

    if save:
        plt.savefig("./save/losslist/loss_list.png")
    plt.close()


if __name__ == "__main__":
    draw_loss_list()
