import pandas as pd

def process_files(file_a, file_b, file_c, output_file):
    df_a = pd.read_csv(file_a, sep=' ', header=None)
    df_b = pd.read_csv(file_b, sep=' ', header=None)
    df_c = pd.read_csv(file_c, sep=' ', header=None)
    
    columns = ['image'] + [f'category_{i}' for i in range(1, 6)]
    df_a.columns = columns
    df_b.columns = columns
    df_c.columns = columns
    
    final_results = {}

    for index, row in df_a.iterrows():
        image = row['image']
        top_1_a = row['category_1']
        
        if top_1_a >= 374 or top_1_a >= 143 and top_1_a <= 243:
            final_results[image] = row.tolist()
        else:
            row_b = df_b[df_b['image'] == image].iloc[0]
            if not row_b.empty:
                top_1_b = row_b['category_1']
                if top_1_b <= 243:
                    final_results[image] = row_b.tolist()
                else:
                    row_c = df_c[df_c['image'] == image].iloc[0]
                    if not row_c.empty:
                        row_c = row_c.tolist()
                        row_c[1:6] = [x + 244 for x in row_c[1:6]]
                        final_results[image] = row_c
                        if top_1_b > 374:
                            print(f"Please check image {image}")
                    else:
                        raise Exception(f"{image} not found in the file!!!")
            else:
                raise Exception(f"{image} not found in the file!!!")

    final_df = pd.DataFrame.from_dict(final_results, orient='index', columns=columns)

    final_df.to_csv(output_file, sep=' ', header=False, index=False)

# process_files('A.txt', 'B.txt', 'C.txt', 'result.txt')
# A clip zero shot, B clip PEFT, C LV-ViT PEFT
