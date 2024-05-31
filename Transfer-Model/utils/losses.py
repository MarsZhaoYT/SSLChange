# from utils.parser import get_parser_with_args
# from options import train_options
from utils.metrics import FocalLoss, dice_loss

# parser = train_options.parser
# opt = parser.parse_args()


def hybrid_loss(prediction, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    # for prediction in predictions:

    bce = focal(prediction, target).cuda()
    dice = dice_loss(prediction, target).cuda()
    loss += bce + dice

    return loss

