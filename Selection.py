from utils import *
def patch_importance(image, patch_size=2, type='variance', how_many=2):
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    H, W = image.shape[-2:]

    value_map = np.zeros((H // patch_size, W // patch_size))

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            start_i = max(i - how_many, 0)
            end_i = min(i + patch_size + how_many, H)
            start_j = max(j - how_many, 0)
            end_j = min(j + patch_size + how_many, W)

            extended_patch = image[start_i:end_i, start_j:end_j]

            if type == 'variance':
                value = np.std(extended_patch)
            elif type == 'mean_brightness':
                value = np.mean(extended_patch)
            elif type == 'contrast':
                value = extended_patch.max() - extended_patch.min()
            elif type == 'edge_density':
                dy, dx = np.gradient(extended_patch)
                value = np.sum(np.sqrt(dx ** 2 + dy ** 2))
            elif type == 'color_diversity':
                value = np.std(extended_patch)

            value_map[i // patch_size, j // patch_size] = value

    return value_map
def chessboard_mask(images, patch_size=1, mask_ratio=0.5, importance_type='variance', how_many=1):
    for mr_i in range(len(mask_ratio)):
        MR = mask_ratio[mr_i]
        B, C, H, W = images.shape
        masked_images = images.clone()
        unmasked_counts = []
        unmasked_patches = []
        patch_index = []

        target_unmasked_ratio = 1 - MR
        num_patches = (H // patch_size) * (W // patch_size)
        target_unmasked_patches = int(num_patches * target_unmasked_ratio)

        for b in range(B):

            patch_importance_map = patch_importance(images[b, 0], patch_size, importance_type, how_many)

            mask = np.zeros((H // patch_size, W // patch_size), dtype=bool)
            for i in range(H // patch_size):
                for j in range(W // patch_size):
                    if (i + j) % 2 == 0:
                        mask[i, j] = True

            unmasked_count = np.sum(~mask)

            if MR < 0.5:
                masked_indices = np.argwhere(mask)
                importances = patch_importance_map[mask]
                sorted_indices = masked_indices[np.argsort(importances)[::-1]]

                for idx in sorted_indices:
                    if unmasked_count >= target_unmasked_patches:
                        break
                    mask[tuple(idx)] = False
                    unmasked_count += 1

            elif MR > 0.5:
                unmasked_indices = np.argwhere(~mask)
                importances = patch_importance_map[~mask]
                sorted_indices = unmasked_indices[np.argsort(importances)]

                for idx in sorted_indices:
                    if unmasked_count <= target_unmasked_patches:
                        break
                    mask[tuple(idx)] = True
                    unmasked_count -= 1

            patches = []
            for i in range(H // patch_size):
                for j in range(W // patch_size):
                    if mask[i, j]:
                        masked_images[b, :, i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size] = 0
                    else:
                        patch = images[b, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

                        patches.append(patch)
                        patch_index.append((H // patch_size) * i + j)

            unmasked_patches.append(torch.cat(patches, dim=-1))
            unmasked_counts.append(unmasked_count)
            unmasked_patches_image = torch.cat(unmasked_patches, dim=-1)

            if MR == 0.234375:
                split_len = 28
                split_tensor = torch.split(unmasked_patches_image, split_len, dim=2)
                masked_23 = masked_images
                reshaped_23 = torch.cat(split_tensor, dim=1)
                index_23 = torch.tensor(patch_index)

            elif MR == 0.4375:
                split_len = 24
                split_tensor = torch.split(unmasked_patches_image, split_len, dim=2)
                masked_43 = masked_images
                reshaped_43 = torch.cat(split_tensor, dim=1)
                index_43 = torch.tensor(patch_index)

            elif MR == 0.609375:
                split_len = 20
                split_tensor = torch.split(unmasked_patches_image, split_len, dim=2)
                masked_60 = masked_images
                reshaped_60 = torch.cat(split_tensor, dim=1)
                index_60 = torch.tensor(patch_index)

    return masked_23, masked_43, masked_60, reshaped_23, reshaped_43, reshaped_60, index_23, index_43, index_60
def preprocess_and_save_dataset(dataset, root_dir, patch_size, mask_ratio, importance_type, how_many):
    os.makedirs(root_dir, exist_ok=True)

    for i, (images, _) in tqdm(enumerate(dataset), total=len(dataset)):

        masked_23, masked_43, masked_60, reshaped_23, reshaped_43, reshaped_60, index_23,  index_43, index_60\
            = chessboard_mask(images.unsqueeze(0), patch_size, mask_ratio, importance_type, how_many)

        torch.save({
            'original_images': images,
            'masked_images_23': masked_23.squeeze(0),
            'masked_images_43': masked_43.squeeze(0),
            'masked_images_60': masked_60.squeeze(0),
            'reshaped_23': reshaped_23,
            'reshaped_43': reshaped_43,
            'reshaped_60': reshaped_60,
            'index_23': index_23,
            'index_43': index_43,
            'index_60': index_60,
        }, os.path.join(root_dir, f'data_{i}.pt'))



class Loader_maker_for_Transmission(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)

        original_images = data['original_images']

        masked_images_23 = data['masked_images_23']
        masked_images_43 = data['masked_images_43']
        masked_images_60 = data['masked_images_60']

        reshaped_23 = data['reshaped_23']
        reshaped_43 = data['reshaped_43']
        reshaped_60 = data['reshaped_60']

        index_23 = data['index_23']
        index_43 = data['index_43']
        index_60 = data['index_60']

        if self.transform:
            original_images = self.transform(original_images)
            masked_images_23 = self.transform(masked_images_23)
            masked_images_43 = self.transform(masked_images_43)
            masked_images_60 = self.transform(masked_images_60)

            reshaped_23 = self.transform(reshaped_23)
            reshaped_43 = self.transform(reshaped_43)
            reshaped_60 = self.transform(reshaped_60)

            index_23 = self.transform(index_23)
            index_43 = self.transform(index_43)
            index_60 = self.transform(index_60)

        return {
            'original_images': original_images,
            'masked_images_23': masked_images_23,
            'masked_images_43': masked_images_43,
            'masked_images_60': masked_images_60,
            'reshaped_23': reshaped_23,
            'reshaped_43': reshaped_43,
            'reshaped_60': reshaped_60,
            'index_23': index_23,
            'index_43': index_43,
            'index_60': index_60
        }
