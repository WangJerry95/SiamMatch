# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--template_path', default='../data/test/20190821/1-SAR/003rel_digs1.bmp', help='datasets')
parser.add_argument('--search_path', default='../data/test/20190821/1-SAR/003ref.bmp', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    # img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # ims = [cv2.imread(imf) for imf in img_files]
    ims = [cv2.imread(args.template_path), cv2.imread(args.search_path)]
    # Select ROI
    cv2.namedWindow("SiamMask")
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # try:
    #     init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
    #     x, y, w, h = init_rect
    # except:
    #     exit()

    template_shape = ims[0].shape
    search_shape = ims[1].shape

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            w = template_shape[1]
            h = template_shape[0]
            target_pos = np.array([search_shape[1] / 2, search_shape[0] / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, search_shape, target_pos, target_sz, siammask, cfg['hp'], device=device)   # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device, debug=True)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            #im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            cv2.imwrite("003rel_digs3.jpg", im)
            key = cv2.waitKey(0)
            if key == 32:
                key = cv2.waitKey(0)
                if key == 32:
                    continue
                elif key > 0:
                    break
            elif key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
