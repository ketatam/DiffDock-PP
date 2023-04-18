from main import main
from args import parse_args
from utils import printt


def main_tune():
    args = parse_args()

    if args.tuned_param == 'lr':
        for lr in [1.e-3, 5.e-3]:
            args.lr = lr
            args.run_name += '_' + args.tuned_param + '_' + str(lr)
            printt(f'run_name: {args.run_name}')
            main(args)

    elif args.tuned_param == 'sigma':
        for sigmas in [ [0.001, 30., 0.001, 1.65], [0.01, 20., 0.01, 1.] ]:
            args.tr_s_min = sigmas[0]
            args.tr_s_max = sigmas[1]
            args.rot_s_min = sigmas[2]
            args.rot_s_max = sigmas[3]

            args.run_name += '_' + args.tuned_param + '_' + str(sigmas[1])
            printt(f'run_name: {args.run_name}')
            main(args)

    elif args.tuned_param == 'cross_cutoff':
        for weight_bias in [ [3, 60], [6, 40] ]:
            args.cross_cutoff_weight = weight_bias[0]
            args.cross_cutoff_bias = weight_bias[1]

            args.run_name += '_' + args.tuned_param + '_' + str(weight_bias[1])
            printt(f'run_name: {args.run_name}')
            main(args)

    elif args.tuned_param == 'tr_rot_weight':
        for tr_rot_weight in [ [0.25, 0.75], [0.75, 0.25] ]:
            args.tr_weight = tr_rot_weight[0]
            args.rot_weight = tr_rot_weight[1]

            args.run_name += '_' + args.tuned_param + '_' + str(tr_rot_weight[1])
            printt(f'run_name: {args.run_name}')
            main(args)


if __name__ == "__main__":
    main_tune()
