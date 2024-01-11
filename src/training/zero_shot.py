import os
import json
import logging

import torch
import numpy as np
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, IMAGENET_FOLDERS2CLASSNAMES, \
    OPENAI_IMAGENET_TEMPLATES, SINGLE_IMAGENET_TEMPLATE
from .precision import get_autocast


def class_top1_accuracy_and_count(
    output: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> (np.ndarray, np.ndarray):
    """
    Calculate top-1 accuracy and count for each class.

    Args:
    output (torch.Tensor): The output tensor from a model, should be logits or probabilities.
    target (torch.Tensor): The ground truth labels, e.g. `torch.tensor([1, 0, 3, 0, 2])`.
    num_classes (int): Number of classes in the dataset.

    Returns:
    tuple: A tuple containing two numpy arrays. The first array contains the top-1 accuracy for each class, 
    and the second array contains the count of samples for each class.
    """
    # # legacy code
    # pred = output.topk(1, 1, True, True)[1].t().cpu().numpy()
    # target = target.cpu().numpy()
    # class_acc1 = np.zeros(num_classes)
    # class_count = np.zeros(num_classes)
    # for i in range(num_classes):
    #     class_acc1[i] = ((pred == i) & (target == i)).sum()
    #     class_count[i] = np.count_nonzero(target == i)
    # return class_acc1, class_count

    pred = output.topk(1, 1, True, True)[1].t()
    class_acc1 = torch.zeros(num_classes, device=output.device)
    class_count = torch.zeros(num_classes, device=output.device)

    for i in range(num_classes):
        class_acc1[i] = ((pred == i) & (target == i)).sum()

    class_count = (target.unsqueeze(0) == torch.arange(
        num_classes, device=output.device).unsqueeze(1)).sum(dim=1)

    return class_acc1.cpu().numpy(), class_count.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        if args.cal_class_top1:
            class_top1 = np.zeros(classifier.shape[1])
            class_n = np.zeros(classifier.shape[1])
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            if args.cal_class_top1:
                class_acc1, class_count = class_top1_accuracy_and_count(
                    logits, target, classifier.shape[1]
                )
                class_top1 += class_acc1
                class_n += class_count

    top1 = (top1 / n)
    top5 = (top5 / n)
    if args.cal_class_top1:
        class_top1 = (class_top1 / class_n)
        return top1, top5, class_top1
    return top1, top5


def extract_certain_text_embeds(
        model,
        classnames,
        templates,
        data_name,
        text_embeds_path,
        args,
        tokenizer
):
    logging.info(f'Extracting text embeds for {data_name}.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            num_classes_per_batch=1,
            device=args.device,
            use_tqdm=True,
            extract_text_embeds=True  # Set to True to extract text embeds
        )
    torch.save(classifier, text_embeds_path)
    logging.info(f'The shape of {text_embeds_path} is {classifier.shape}.')
    logging.info(f'{text_embeds_path} saved.')


def extract_text_embeds(model, data, args, tokenizer):

    text_embeds_root = '../data/ImageNet/text_embeds'
    if not os.path.exists(text_embeds_root):
        os.makedirs(text_embeds_root)
    text_embeds_model = args.model.replace(
        '-text-embeds-extractor', ''
    ).replace('-', '_')
    logging.info(f'Start extracting text embeds with {args.model}.')

    if 'imagenet-overall-prompt' in data:
        data_name = 'imagenet-overall-prompt'.replace('-', '_')
        with open(f'../prompts/{data_name}.json', 'r') as f:
            prompt_dict = json.load(f)
        classnames = [
            prompt_dict[classname][0]
            for classname in IMAGENET_CLASSNAMES
        ]
        text_embeds_path = f'{text_embeds_root}/{data_name}_{text_embeds_model}.pt'
        extract_certain_text_embeds(
            model,
            classnames=classnames,
            templates=(lambda c: c, ),
            data_name=data_name,
            text_embeds_path=text_embeds_path,
            args=args,
            tokenizer=tokenizer
        )

    if 'imagenet-single-template' in data:
        data_name = 'imagenet-single-template'.replace('-', '_')
        text_embeds_path = f'{text_embeds_root}/{data_name}_{text_embeds_model}.pt'
        extract_certain_text_embeds(
            model,
            classnames=IMAGENET_CLASSNAMES,
            templates=SINGLE_IMAGENET_TEMPLATE,
            data_name=data_name,
            text_embeds_path=text_embeds_path,
            args=args,
            tokenizer=tokenizer
        )

    logging.info(f'Finished extracting text embeds with {args.model}.')


def eval_for_training_with_text_embeds(model, data, args):
    logging.info(
        f'Start evaluating {args.model} trained with {args.text_embeds_path}.'
    )
    results = {}

    def get_eval_text_embeds_path(data_name):
        # args.text_embeds_path is the path of text embeds for training,
        # need to get text_embeds_path corresponding to each dataset
        text_embeds_root = '../data/ImageNet/text_embeds'

        if 'RN50' in args.model:
            text_embeds_model = args.model.replace(
                '-RN50', ''
            ).replace('-', '_')
        else:
            raise ValueError(f'Model with unknown vision tower {args.model}.')

        if data_name == 'imagenet_val':
            # FIXME NOT SUITABLE in function zero_shot_eval()
            # When evaluating models trained with extracted text embeddings,
            # imagenet-val is used as the traditional iid validation set,
            # other than used for zero-shot evaluation,
            # so return the same text_embeds_path as training
            return args.text_embeds_path
        else:
            eval_text_embeds_path = f'{text_embeds_root}/{data_name}_{text_embeds_model}.pt'
            if os.path.exists(eval_text_embeds_path):
                return eval_text_embeds_path
            else:
                raise ValueError(
                    f'Text embeddings not found: {eval_text_embeds_path}')

    def get_eval_class_index_and_names(root):

        if "ImageNet" in root:
            all_folders = sorted(IMAGENET_FOLDERS2CLASSNAMES.keys())
            all_classnames = IMAGENET_CLASSNAMES
        else:
            raise ValueError(f"Unsupported data folder: {root}.")

        eval_folders = sorted(
            [
                folder for folder in os.listdir(root)
                if os.path.isdir(os.path.join(root, folder))
            ]
        )

        eval_class_index = [
            all_folders.index(folder) for folder in eval_folders
        ]
        eval_classnames = [
            all_classnames[index] for index in eval_class_index
        ]

        return eval_class_index, eval_classnames

    def build_eval_classifier_with_text_embeds(
        model,
        data_name,
        data_folder,
        templates
    ):
        logging.info(f'Building eval classifier for {data_name}.')
        eval_text_embeds_path = get_eval_text_embeds_path(data_name)
        eval_class_index, eval_classnames = get_eval_class_index_and_names(
            data_folder
        )

        autocast = get_autocast(args.precision)
        with autocast():
            with torch.no_grad():
                eval_classifier = torch.load(
                    eval_text_embeds_path
                ).T[eval_class_index]
                # For the text_embeds_projection in text tower
                eval_classifier = model.encode_text(
                    eval_classifier,
                    normalize=True
                )
                eval_classifier = eval_classifier.reshape(
                    len(eval_class_index),
                    len(templates),
                    -1
                ).mean(dim=1)
                eval_classifier = eval_classifier / \
                    eval_classifier.norm(dim=1, keepdim=True)
                eval_classifier = eval_classifier.T

        return eval_classifier, eval_classnames

    if 'imagenet-overall-prompt' in data:
        data_name = 'imagenet-overall-prompt'.replace('-', '_')
        data_folder = args.imagenet_overall_prompt
        eval_classifier, eval_classnames = build_eval_classifier_with_text_embeds(
            model,
            data_name,
            data_folder,
            templates=(lambda c: c, )
        )
        logging.info(f'Using eval classifier for {data_name}.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-overall-prompt'].dataloader,
            args
        )
        results['imagenet-overall-prompt-top1'] = metrics[0]
        results['imagenet-overall-prompt-top5'] = metrics[1]
        if args.cal_class_top1:
            class_top1 = dict(zip(eval_classnames, metrics[2]))
            results['imagenet-overall-prompt-class-top1'] = sorted(
                class_top1.items(), key=lambda x: x[1], reverse=True
            )

    if 'imagenet-single-template' in data:
        data_name = 'imagenet-single-template'.replace('-', '_')
        data_folder = args.imagenet_single_template
        eval_classifier, eval_classnames = build_eval_classifier_with_text_embeds(
            model,
            data_name,
            data_folder,
            templates=SINGLE_IMAGENET_TEMPLATE
        )
        logging.info(f'Using eval classifier for {data_name}.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-single-template'].dataloader,
            args
        )
        results['imagenet-single-template-top1'] = metrics[0]
        results['imagenet-single-template-top5'] = metrics[1]
        if args.cal_class_top1:
            class_top1 = dict(zip(eval_classnames, metrics[2]))
            results['imagenet-single-template-class-top1'] = sorted(
                class_top1.items(), key=lambda x: x[1], reverse=True
            )

    if 'imagenet-val' in data:
        data_name = 'imagenet_val'
        data_folder = args.imagenet_val
        eval_classifier, eval_classnames = build_eval_classifier_with_text_embeds(
            model,
            data_name,
            data_folder,
            # FIXME templates should be the same as training, but static here
            templates=(lambda c: c, )
        )
        logging.info(f'Using eval classifier for {data_name}.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-val'].dataloader,
            args
        )
        results['imagenet-val-top1'] = metrics[0]
        results['imagenet-val-top5'] = metrics[1]
        if args.cal_class_top1:
            class_top1 = dict(zip(eval_classnames, metrics[2]))
            results['imagenet-val-class-top1'] = sorted(
                class_top1.items(), key=lambda x: x[1], reverse=True
            )

    logging.info(
        f'Finished evaluating {args.model} trained with {args.text_embeds_path}.')
    return results


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    zero_shot_eval_dataset_names = [
        'imagenet-val',
        'imagenet-v2',
        'imagenet-overall-prompt',
        'imagenet-single-template'
    ]
    if not any([dataset_name in data for dataset_name in zero_shot_eval_dataset_names]):
        logging.info('No zero-shot evaluation datasets found.')
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    # For extracting text embeddings with frozen LLM
    if 'text-embeds-extractor' in args.model:
        extract_text_embeds(model, data, args, tokenizer)
        return {}

    # For evaluation after training with extracted text embeddings
    if args.text_embeds_path is not None:
        return eval_for_training_with_text_embeds(model, data, args)

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
