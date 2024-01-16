import os
import json
import logging

import torch
import numpy as np
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, IMAGENET_FOLDERS2CLASSNAMES, \
    OPENAI_IMAGENET_TEMPLATES, SINGLE_IMAGENET_TEMPLATE, SIMPLE_IMAGENET_TEMPLATES
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


def get_templates_from_text_type(text_type):
    template_dict = {
        "imagenet_simple_templates": SIMPLE_IMAGENET_TEMPLATES,
        "imagenet_openai_templates": OPENAI_IMAGENET_TEMPLATES,
        "imagenet_single_template": SINGLE_IMAGENET_TEMPLATE,
        # FIXME this is a hack for text_type prompt
        "imagenet_overall_prompt": (lambda c: c, ),
        "imagenet_points_prompt": (lambda c: c, )
    }
    if text_type not in template_dict:
        raise ValueError(f"Unsupported text type: {text_type}.")
    return template_dict[text_type]


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
    
    if 'imagenet-points-prompt' in data:
        # First step: merge all points for each class
        data_name = 'imagenet-points-prompt'.replace('-', '_')
        text_embeds_path = f'{text_embeds_root}/{data_name}_{text_embeds_model}.pt'
        with open(f'../prompts/{data_name}.json', 'r') as f:
            prompt_dict = json.load(f)
        # The sort of classnames in imagenet_points_prompt.json has been checked to be correct
        classnames = [
            f'This is a photo of the {classname}, some typical visual features include: ' + ','.join(prompt_list) + '.'
            for classname, prompt_list in prompt_dict.items()
        ]
        extract_certain_text_embeds(
            model,
            classnames=classnames,
            templates=(lambda c: c, ),
            data_name=data_name,
            text_embeds_path=text_embeds_path,
            args=args,
            tokenizer=tokenizer
        )
        # Second step: extract text embeds for each point separately
        separate_data_name = 'imagenet-points-prompt-separate'.replace('-', '_')
        separate_text_embeds_folder = f'{text_embeds_root}/{separate_data_name}_{text_embeds_model}'
        if not os.path.exists(separate_text_embeds_folder):
            os.makedirs(separate_text_embeds_folder)
        separate_classnames = [
            (f'This is a photo of the {classname}.', *prompt_list)
            for classname, prompt_list in prompt_dict.items()
        ]
        for i, separate_classname in enumerate(separate_classnames):
            separate_text_embeds_path = f'{separate_text_embeds_folder}/{i}.pt'
            extract_certain_text_embeds(
                model,
                classnames=separate_classname,  # a tuple
                templates=(lambda c: c, ),
                data_name=f'{separate_data_name}_class_{i}',
                text_embeds_path=separate_text_embeds_path,
                args=args,
                tokenizer=tokenizer
            )

    logging.info(f'Finished extracting text embeds with {args.model}.')


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


def eval_for_training_with_text_embeds(model, data, args):
    logging.info(
        f'Start evaluating {args.model} trained with {args.text_embeds_path}.'
    )
    results = {}

    def get_eval_text_embeds_path(data_name, separate_per_class=False):
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
        elif separate_per_class:
            eval_text_embeds_folder = f'{text_embeds_root}/{data_name}_{text_embeds_model}'
            if os.path.exists(eval_text_embeds_folder):
                return eval_text_embeds_folder
            else:
                raise ValueError(
                    f'Text embeddings not found: {eval_text_embeds_folder}')
        else:
            eval_text_embeds_path = f'{text_embeds_root}/{data_name}_{text_embeds_model}.pt'
            if os.path.exists(eval_text_embeds_path):
                return eval_text_embeds_path
            else:
                raise ValueError(
                    f'Text embeddings not found: {eval_text_embeds_path}')

    def build_eval_classifier_with_text_embeds(
        model,
        data_name,
        data_folder,
        templates,
        separate_per_class=False
    ):
        logging.info(f'Building eval classifier for {data_name}.')

        if separate_per_class:
            eval_text_embeds_folder = get_eval_text_embeds_path(
                data_name, separate_per_class=separate_per_class
            )
        else:
            eval_text_embeds_path = get_eval_text_embeds_path(data_name)

        eval_class_index, eval_classnames = get_eval_class_index_and_names(
            data_folder
        )

        autocast = get_autocast(args.precision)
        with autocast():
            with torch.no_grad():

                if separate_per_class:
                    all_separate_class_embeds = [
                        torch.load(
                            os.path.join(eval_text_embeds_folder, f'{i}.pt')
                        ).T
                        for i in eval_class_index
                    ]
                    for i, separate_class_embeds in enumerate(all_separate_class_embeds):
                        separate_class_embeds = model.encode_text(
                            separate_class_embeds,
                            normalize=True
                        )
                        all_separate_class_embeds[i] = separate_class_embeds[0] * \
                            0.5 + separate_class_embeds[1:].mean(dim=0) * 0.5
                    eval_classifier = torch.stack(
                        all_separate_class_embeds, dim=0
                    )
                else:
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

    if 'imagenet-points-prompt' in data:
        data_name = 'imagenet-points-prompt'.replace('-', '_')
        data_folder = args.imagenet_points_prompt
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
            data['imagenet-points-prompt'].dataloader,
            args
        )
        results['imagenet-points-prompt-top1'] = metrics[0]
        results['imagenet-points-prompt-top5'] = metrics[1]
        if args.cal_class_top1:
            class_top1 = dict(zip(eval_classnames, metrics[2]))
            results['imagenet-points-prompt-class-top1'] = sorted(
                class_top1.items(), key=lambda x: x[1], reverse=True
            )

    if 'imagenet-points-prompt' in data and args.eval_points_prompt_separately:
        separate_data_name = 'imagenet-points-prompt-separate'.replace(
            '-', '_')
        separate_data_folder = args.imagenet_points_prompt
        eval_classifier, eval_classnames = build_eval_classifier_with_text_embeds(
            model,
            separate_data_name,
            separate_data_folder,
            templates=(lambda c: c, ),
            separate_per_class=True
        )
        logging.info(f'Using eval classifier for {separate_data_name}.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-points-prompt'].dataloader,
            args
        )
        results['imagenet-points-prompt-separate-top1'] = metrics[0]
        results['imagenet-points-prompt-separate-top5'] = metrics[1]
        if args.cal_class_top1:
            class_top1 = dict(zip(eval_classnames, metrics[2]))
            results['imagenet-points-prompt-separate-class-top1'] = sorted(
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
            templates=get_templates_from_text_type(args.text_type)
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


def zero_shot_eval_model_trained_from_scratch(model, data, args, tokenizer):
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info(f"Starting evaluating {args.model} trained from scratch.")
    results = {}

    if 'imagenet-val' in data:
        eval_classnames = get_eval_class_index_and_names(args.imagenet_val)[1]
        logging.info(f'Building eval classifier for imagenet_val.')
        autocast = get_autocast(args.precision)
        with autocast():
            eval_classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=eval_classnames,
                templates=get_templates_from_text_type(args.text_type),
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
        logging.info(f'Using eval classifier for imagenet_val.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-val'].dataloader,
            args
        )
        results['imagenet-val-top1'] = metrics[0]
        results['imagenet-val-top5'] = metrics[1]

    if 'imagenet-zero-shot' in data:
        eval_classnames = get_eval_class_index_and_names(
            args.imagenet_zero_shot)[1]
        logging.info(f'Building eval classifier for imagenet-zero-shot.')
        autocast = get_autocast(args.precision)
        with autocast():
            eval_classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=eval_classnames,
                templates=get_templates_from_text_type(args.text_type),
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
        logging.info(f'Using eval classifier for imagenet-zero-shot.')
        metrics = run(
            model,
            eval_classifier,
            data['imagenet-zero-shot'].dataloader,
            args
        )
        results['imagenet-zero-shot-top1'] = metrics[0]
        results['imagenet-zero-shot-top5'] = metrics[1]

    logging.info(f"Finished evaluating {args.model} trained from scratch.")
    return results


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    zero_shot_eval_dataset_names = [
        'imagenet-val',
        'imagenet-v2',
        'imagenet-overall-prompt',
        'imagenet-points-prompt',
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

    if "ImageNet" in args.train_data:
        return zero_shot_eval_model_trained_from_scratch(model, data, args, tokenizer)

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
