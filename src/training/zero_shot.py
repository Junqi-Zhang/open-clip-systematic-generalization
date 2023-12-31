import json
import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SINGLE_IMAGENET_TEMPLATES
from .precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
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

    top1 = (top1 / n)
    top5 = (top5 / n)
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
            templates=SINGLE_IMAGENET_TEMPLATES,
            data_name=data_name,
            text_embeds_path=text_embeds_path,
            args=args,
            tokenizer=tokenizer
        )

    logging.info(f'Finished extracting text embeds with {args.model}.')


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

    # For extracting text embeds with frozen LLM
    if 'text-embeds-extractor' in args.model:
        extract_text_embeds(model, data, args, tokenizer)
        return {}

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
