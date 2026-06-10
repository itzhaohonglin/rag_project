# 1. 导入所需的库
from pymilvus import MilvusClient, DataType

# ================= 第一部分：连接到 Milvus =================
print("正在连接 Milvus...")
# 使用 MilvusClient 连接本地 Docker 服务 (默认端口19530)
# 也可换成本地文件 "./milvus_demo.db" 来使用 Milvus Lite
client = MilvusClient(uri="http://localhost:19530")
# 如果服务开启了认证，需添加 token="root:Milvus" 等参数[reference:4]
print("✅ 连接成功！\n")
#
# # ================= 第二部分：创建集合 (Create) =================
collection_name = "example_collection"
# 如果集合已存在，先删除旧的，确保脚本可重复运行
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"已删除已有集合：{collection_name}")
#
# 定义集合的 Schema (结构)
# 包含三个字段：id (主键), vector (向量), author (作者，用于标量过滤)
# schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
# 1. 添加 id 字段，类型为 INT64，并设为主键
# schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
# 2. 添加 vector 字段，类型为浮点型向量，维度设为 128
# schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
# 3. 添加 author 字段，类型为 VARCHAR，最大长度 100 字符
# schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=100)

# 创建集合
# client.create_collection(
#     collection_name=collection_name,
#     schema=schema,
#     metric_type="L2",          # 指定向量距离算法为欧氏距离
#     vector_field_name="vector" # 指定哪个字段是向量字段
# )
# print(f"✅ 成功创建集合：{collection_name}\n")

# 建索引（Milvus 加载集合前必须有索引）
# index_params = client.prepare_index_params()
# index_params.add_index(
#     field_name="vector",
#     index_type="IVF_FLAT",
#     metric_type="L2",
#     params={"nlist": 128}
# )
# client.create_index(collection_name, index_params)
# print("✅ 索引创建成功\n")

# ================= 第三部分：插入数据 (Create) =================
# 准备要插入的数据，每条数据是一个字典，包含 id, vector, author 三个字段
# data_to_insert = [
#     {"vector": [0.1] * 128, "author": "Alice"},   # 简单演示，用相同数值填充
#     {"vector": [0.2] * 128, "author": "Bob"},
#     {"vector": [0.3] * 128, "author": "Charlie"},
# ]
# insert_result = client.insert(collection_name, data_to_insert)
# print(f"✅ 数据插入结果: {insert_result}")  # 返回值会包含插入数量等
# print(f"✅ 成功插入 {insert_result['insert_count']} 条数据\n")
# # 记下自动生成的 ID，后面按 ID 查询用
# generated_ids = insert_result["ids"]
# print(f"自动生成的 ID: {generated_ids}\n")

# # ================= 第四部分：查询数据 (Read) =================
# 查询前需要加载集合到内存
client.load_collection(collection_name)
print("=== 通过标量条件查询 ===")
# 场景1：查询作者为 'Bob' 的所有数据
res_query_by_filter = client.query(
    collection_name=collection_name,
    filter="author == 'Bob'",   # 过滤表达式
    output_fields=["author", "vector"] # 指定返回哪些字段
)
print("查询条件 author == 'Bob':", res_query_by_filter)

print("\n=== 通过主键 ID 查询 ===")
# 场景2：根据主键 ID 精确查询
res_query_by_id = client.query(
    collection_name=collection_name,
    # ids=generated_ids[0],
    ids=[466836994310802383],
    output_fields=["author"]
)
print("查询所有 ID 的作者:", res_query_by_id)
print(f"查询到 {len(res_query_by_id)} 条记录\n")

# ================= 第五部分：更新/替换数据 (Update) =================
# print("=== 更新数据 ===")
# # 注意：Milvus 的更新本质是“按主键替换”，即插入同样主键的数据
# # 准备要更新的数据，主键 id 必须与现有数据相同
# data_to_upsert = [
#     {"id": 466836994310802383, "vector": [0.25] * 128, "author": "Robert"} # 将 Bob 的 author 改为 Robert
# ]
# # 使用 upsert 方法，如果主键存在则更新，不存在则插入
# update_result = client.upsert(collection_name, data_to_upsert)
# print(f"✅ 更新结果: {update_result}")
#
# # 验证更新是否成功
# res_after_update = client.query(collection_name, filter="author == 'Robert'")
# print("更新后查询 author == 'Robert':", res_after_update, "\n")

# # ================= 第六部分：向量相似度搜索 =================
print("=== 向量相似度搜索 ===")
# 准备一个查询向量，类型、维度必须与库内数据一致
query_vector = [0.3] * 128

# 执行搜索，查找与 query_vector 最相似的 2 条记录
search_results = client.search(
    collection_name=collection_name,
    data=[query_vector],        # 待查询的向量列表，支持批量
    anns_field="vector",        # 指定在哪个向量字段上搜索
    limit=2,                    # 返回最相似的 2 条记录
    output_fields=["author"]    # 返回结果时附带 author 字段
)

print(f"查询向量 {query_vector[:3]}... 的搜索结果:")
for i, result in enumerate(search_results[0]):  # search_results[0] 是第一条搜索的结果
    print(f"  第 {i+1} 个结果: ID={result['id']}, 作者={result['entity']['author']}, 距离={result['distance']:.4f}")
print()
#
# # ================= 第七部分：删除数据 (Delete) =================
# print("=== 删除数据 ===")
# # 删除指定主键 ID 的数据
# delete_result = client.delete(collection_name, ids=[1])
# print(f"✅ 删除结果: {delete_result}")
#
# # 验证删除是否成功
# remaining_data = client.query(collection_name, output_fields=["id"])
# print(f"删除后剩余的数据 ID: {[item['id'] for item in remaining_data]}\n")
#
# # ================= 第八部分：清理资源 =================
# print("=== 清理资源 ===")
# # 如果不再需要，可以删除整个集合
# # client.drop_collection(collection_name)
# # print(f"已删除集合: {collection_name}")
# print("脚本执行完毕。")