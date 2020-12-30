# ブログ


## このブログで理解しとくこと
- 基本はjekyll
- themeはjekyll-theme-clean-blog
  - ただし,themeで指定するのではなくローカルにコピーしてカスタマイズして利用している.
  - これにした理由はbootstrapを使って癖のないフォーマットだったから.
- scssはcssに手動コンパイルする必要がある.
  ```
  sass assets/main.scss assets/main.css --trace
  ```


## TODO
- [ ] main.cssの自動更新
  - gulpを真面目に使えばできそうだけど、そこまでする必要はないと判断.
- [ ] esaとjekyllの連携
  - https://blog.waft.me/2016/09/19/esa-circl-ci/
- [x] 右サイドバーの作成
- [ ] 右サイドバーのコンテンツ拡充
  - [x] タグごとの切り分け
  - [x] twitter/githubのアカウント
  - [ ] 見た目の改善は必要(sidebar.html)
    - タグ自体と段落の間の改行
    - 外部アカウントとの改行
    - 見る媒体ごとの違いはある?
- [x] GAの設定
  - https://qiita.com/memakura/items/2cfc8133e07fdc72c45f
- [ ] タグページの改良
  - リクエストで関係ないタグを削除する?
  - もしくは純粋に長い記事リストとして紹介する?
  - 本来はDBを使うべき内容.(tagとURLのリンクを動的に作成したい.)
- [ ] 分野ごとのリンク作成
  - これはある程度ちゃんと作ってから
- [x] 上部の空間の減少
  - 改善はしたものの、まだ課題あり
- [x] h2,h1に対する改善
- [x] 数式あり
  - https://sekika.github.io/2015/10/10/equation-on-jekyll/
- [ ] 全体的なグレードアップ
  - https://masamichi.me/ を参考に
- [x] google analyticsとSEO対策作成
- [x] 数学のページも作成
  - PDF置き場