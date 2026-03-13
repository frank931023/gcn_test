"""
缓存管理工具 - 查看、清理、强制重新计算
"""
import os
import sys


def check_cache():
    """检查缓存状态"""
    print("\n" + "="*60)
    print("📦 缓存检查")
    print("="*60)
    
    cache_files = [
        'outputs/standardized_transactions_cached.csv',
        'outputs/account_to_id.pt'
    ]
    
    all_exist = True
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            size = os.path.getsize(cache_file) / 1024 / 1024  # 转换为MB
            print(f"✓ 存在: {cache_file} ({size:.2f} MB)")
        else:
            print(f"✗ 缺失: {cache_file}")
            all_exist = False
    
    if all_exist:
        print("\n状态: ✅ 缓存完整，可直接使用")
    else:
        print("\n状态: ⚠️  缓存不完整，下次运行会重新计算")
    
    return all_exist


def clear_cache():
    """清除缓存"""
    print("\n" + "="*60)
    print("🗑️  清除缓存")
    print("="*60)
    
    cache_files = [
        'outputs/standardized_transactions_cached.csv',
        'outputs/account_to_id.pt'
    ]
    
    removed = 0
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"✓ 已删除: {cache_file}")
                removed += 1
            except Exception as e:
                print(f"✗ 删除失败: {cache_file} ({str(e)})")
        else:
            print(f"  跳过: {cache_file} (不存在)")
    
    if removed > 0:
        print(f"\n✅ 已清除 {removed} 个缓存文件。下次运行会重新计算。")
    else:
        print(f"\n⚠️  没有找到缓存文件。")
    
    return removed


def run_with_cache():
    """使用缓存运行（如果存在）"""
    print("\n使用缓存运行 (如果缓存存在)...")
    os.system('cd gcn && python -c "from main import main; main(use_cache=True)"')


def run_without_cache():
    """强制重新计算，不使用缓存"""
    print("\n强制重新计算 (清除缓存后运行)...")
    clear_cache()
    run_with_cache()


def main_menu():
    """主菜单"""
    while True:
        print("\n" + "="*60)
        print("缓存管理工具")
        print("="*60)
        print("1. 检查缓存状态")
        print("2. 清除缓存")
        print("3. 使用缓存运行")
        print("4. 强制重新计算（清除缓存+重新跑）")
        print("5. 退出")
        print("-"*60)
        
        choice = input("请选择 (1-5): ").strip()
        
        if choice == '1':
            check_cache()
        elif choice == '2':
            confirm = input("\n确定要清除缓存吗? (y/n): ").strip().lower()
            if confirm == 'y':
                clear_cache()
            else:
                print("已取消")
        elif choice == '3':
            run_with_cache()
        elif choice == '4':
            confirm = input("\n确定要重新计算吗? (y/n): ").strip().lower()
            if confirm == 'y':
                run_without_cache()
            else:
                print("已取消")
        elif choice == '5':
            print("退出")
            break
        else:
            print("无效选择，请重试")


if __name__ == '__main__':
    # 如果有命令行参数
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'check':
            check_cache()
        elif cmd == 'clear':
            clear_cache()
        elif cmd == 'run':
            run_with_cache()
        elif cmd == 'reset':
            run_without_cache()
        else:
            print(f"未知命令: {cmd}")
            print("\n可用命令:")
            print("  python cache_tools.py check   - 检查缓存")
            print("  python cache_tools.py clear   - 清除缓存")
            print("  python cache_tools.py run     - 使用缓存运行")
            print("  python cache_tools.py reset   - 强制重新计算")
    else:
        # 没有参数则显示菜单
        main_menu()
